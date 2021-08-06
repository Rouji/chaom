#!/usr/bin/env python3
import re
from glob import glob
from typing import *
from dataclasses import dataclass
from subprocess import PIPE, DEVNULL, Popen
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors.content_detector import ContentDetector
from concurrent import futures
from os import unlink, mkdir
from os.path import join, exists, basename
from pickle import dump, load
from hashlib import md5
from functools import reduce
from multiprocessing import cpu_count
from shutil import move, rmtree
import argparse
from sys import stderr

Timestamp = float

def err(*args, **kwargs):
    kwargs.update({'file': stderr})
    print(*args, **kwargs)

def unlink_ignore(*args, **kwargs):
    try:
        unlink(*args, **kwargs)
    except FileNotFoundError:
        pass

@dataclass(frozen=True)
class Frame:
    nr: int
    time: Timestamp

@dataclass(frozen=True)
class Segment:
    start: Frame
    end: Frame

def iframes(infile: str) -> Optional[List[Frame]]:
    args = [
        'ffmpeg',
        '-hide_banner',
        '-i', infile,
        '-vf', 'select=eq(pict_type\\,PICT_TYPE_I)',
        '-loglevel', 'debug',
        '-f', 'null',
        '-',
    ]
    proc = Popen(args, stdin=DEVNULL, stdout=DEVNULL, stderr=PIPE)

    regex = re.compile(r'n:(?P<nr>\d+)\.0+ pts:\d+\.\d+ t:(?P<time>\d+\.\d+) key:1')
    i = [
        Frame(int(match.group('nr')), Timestamp(match.group('time')))
        for match in (regex.search(line.decode()) for line in (proc.stderr or b''))
        if match
    ]

    return i if proc.wait() == 0 else None

def scene_segments(file: str, threshold: int = 30) -> List[Segment]:
    vm = VideoManager([file])
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold))
    vm.set_downscale_factor()
    vm.start()
    sm.detect_scenes(frame_source=vm, show_progress=False)
    return [
        Segment(
            Frame(start.frame_num, start.get_seconds()),
            Frame(end.previous_frame().frame_num, end.previous_frame().get_seconds())
        )
        for start,end in sm.get_scene_list()
    ]

def merge_segments(segments: List[Segment], min_length: float) -> List[Segment]:
    def duration(segment: Segment) -> float:
        return segment.end.time - segment.start.time
    def merge_if_shorter(s1: List[Segment], s2: List[Segment]) -> List[Segment]:
        if duration(s1[-1]) < min_length or duration(s2[0]) < min_length:
            return s1[:-1] + [Segment(s1[-1].start, s2[0].end)]
        else:
            return s1 + s2

    return reduce(merge_if_shorter, [[s] for s in segments])


def ffmpeg_segment_pipe(infile: str, segment: Segment, seek: Optional[Frame] = None, stderr=None) -> Popen:
    seek_args = ['-ss', str(seek.time)] if seek else []
    seek_nr = seek.nr if seek else 0
    args = [
        'ffmpeg',
        '-y',
        '-hide_banner',
        '-loglevel', 'error',
        ] + seek_args + [
        '-i', infile,
        '-map', '0:v:0',
        '-vf', f'select=between(n\\,{segment.start.nr - seek_nr}\\,{segment.end.nr - seek_nr}),setpts=PTS-STARTPTS',
        '-strict', '-1',
        '-pix_fmt', 'yuv420p10le',
        '-f', 'yuv4mpegpipe', '-',
    ]
    return Popen(args, stdout=PIPE, stderr=stderr)

def aom_encode(ffmpeg_pipe: Popen, args: List[str], outfile: str, stderr=None) -> Popen:
    return Popen(['aomenc'] + args + ['-o', outfile, '-'], stdin=ffmpeg_pipe.stdout, stderr=stderr)

def nearest_iframe(segment: Segment, iframes: List[Frame]) -> Optional[Frame]:
    earlier_iframes = [f for f in iframes if f.time < segment.start.time]
    return earlier_iframes[-1] if earlier_iframes else None

def process_segment(infile: str, seg: Segment, segments_dir: str, aom_args: List[str], seek: Optional[Frame] = None) -> bool:
    seg_name = f'{seg.start.nr:08}-{seg.end.nr:08}'
    fpf_tmp_file = join(segments_dir, f'_{seg_name}.fpf')
    fpf_finished_file = join(segments_dir, f'{seg_name}.fpf')
    seg_tmp_file = join(segments_dir, f'_{seg_name}.ivf')
    seg_finished_file = join(segments_dir, f'{seg_name}.ivf')

    unlink_ignore(fpf_tmp_file)
    unlink_ignore(seg_tmp_file)

    if exists(seg_finished_file):
        return True

    if not exists(seg_finished_file):
        with open(join(segments_dir, f'{seg_name}_pass1_ffmpeg_stderr.log'), 'wb') as fferr, open(join(segments_dir, f'{seg_name}_pass1_aomenc_stderr.log'), 'w') as aomerr:
            ff = ffmpeg_segment_pipe(infile, seg, seek=seek, stderr=fferr)
            aom = aom_encode(ff, ['--passes=2', '--pass=1', f'--fpf={fpf_tmp_file}'] + aom_args, '/dev/null', stderr=aomerr)
            if aom.wait() != 0 or ff.wait() != 0:
                err(f'1st pass of segment {seg_name} failed')
                return False
            else:
                move(fpf_tmp_file, fpf_finished_file)

    with open(join(segments_dir, f'{seg_name}_pass2_ffmpeg_stderr.log'), 'wb') as fferr, open(join(segments_dir, f'{seg_name}_pass2_aomenc_stderr.log'), 'w') as aomerr:
        ff = ffmpeg_segment_pipe(infile, seg, seek=seek, stderr=fferr)
        aom = aom_encode(ff, ['--passes=2', '--pass=2', f'--fpf={fpf_finished_file}'] + aom_args, seg_tmp_file, stderr=aomerr)
        if aom.wait() != 0 or ff.wait() != 0:
            err(f'2nd pass of segment {seg_name} failed')
            unlink_ignore(seg_tmp_file)
            return False

    move(seg_tmp_file, seg_finished_file)
    return True


def concat(segments_dir: str, infile: str, outfile: str, stdout=None, stderr=None) -> bool:
    segments = sorted([basename(f) for f in glob(join(segments_dir, '*.ivf'))])
    concat_file = join(segments_dir, 'concat.txt')
    with open(concat_file, 'w') as f:
        f.write('\n'.join([f"file '{s}'" for s in segments]))
    args = [
        'ffmpeg',
        '-y', '-hide_banner',
        '-loglevel', 'error',
        '-i', infile,
        '-f', 'concat', '-safe', '0', '-i', concat_file,
        '-map_metadata', '0:g',
        '-map', '1:v',
        '-map', '0:a?',
        '-map', '0:s?',
        '-c', 'copy',
        '-y', outfile
    ]
    return Popen(args, stdout=stdout, stderr=stderr).wait() == 0


T = TypeVar('T')
def file_cache(file: str, func: Callable[..., T], *args, **kwargs) -> T:
    if exists(file):
        with open(file, 'rb') as f:
            return load(f)
    else:
        with open(file, 'wb') as f:
            out = func(*args, **kwargs)
            dump(out, f)
            return out

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='file to encode')
    parser.add_argument('output_file', type=str, help='path to encoded output file')
    parser.add_argument('-w', '--workers', type=int, default=None, help='max. worker count')
    parser.add_argument('-s', '--min-segment-length', type=float, default=1.0, help='min. segment length in seconds.')
    parser.add_argument('-t', '--tmpdir', default=None, help='directory to use for temporary files')
    parser.add_argument('-a', '--aom-args', type=str, default='', help='arguments to pass to aomenc')
    parser.add_argument('-k', '--keep', action='store_true', help='keep output segments and other temp data even on success')
    parser.add_argument('--scene-threshold', type=int, default=30, help='scene detection threshold in percent')
    args = parser.parse_args()

    infile: str = args.input_file
    aom_args = args.aom_args.strip().split()
    workers = args.workers or cpu_count()

    tmpdir = args.tmpdir or '.'+md5(infile.encode()).hexdigest()
    try:
        mkdir(tmpdir)
    except FileExistsError:
        pass

    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        iframes_cache_file = join(tmpdir, 'iframes.bin')
        segments_cache_file = join(tmpdir, 'segments.bin')

        iframes_fut = executor.submit(file_cache, iframes_cache_file, iframes, infile)
        segments_fut = executor.submit(file_cache, segments_cache_file, lambda: merge_segments(scene_segments(infile, threshold=args.scene_threshold), args.min_segment_length))

        _iframes: List[Frame] = iframes_fut.result()
        segments: List[Segment] = segments_fut.result()

        if not segments or not _iframes:
            return 1

        fut = [executor.submit(process_segment, infile, seg, tmpdir, aom_args, seek=nearest_iframe(seg, _iframes)) for seg in sorted(segments, key=lambda s: s.end.nr - s.start.nr, reverse=True)]
        res = [f.result() for f in fut]
        if False in res:
            return 1

    with open(join(tmpdir, 'concat_stderr.log'), 'wb') as fferr:
        if not concat(tmpdir, infile, args.output_file, stdout=fferr):
            err('concatenation failed')
            return 1
    if not args.keep:
        rmtree(tmpdir)
    return 0

if __name__ == '__main__':
    exit(main())
