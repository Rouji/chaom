Crappy script for chunked parallel AV1 encoding

```bash
# python3 -m chaom -h                                                                                                                                                                                                                                                                                                                                   chaom [master]
usage: __main__.py [-h] [-w WORKERS] [-s MIN_SEGMENT_LENGTH] [-t TMPDIR] [-a AOM_ARGS] [-k] [--scene-threshold SCENE_THRESHOLD] input_file output_file

positional arguments:
  input_file            file to encode
  output_file           path to encoded output file

optional arguments:
  -h, --help            show this help message and exit
  -w WORKERS, --workers WORKERS
                        max. worker count
  -s MIN_SEGMENT_LENGTH, --min-segment-length MIN_SEGMENT_LENGTH
                        min. segment length in seconds.
  -t TMPDIR, --tmpdir TMPDIR
                        directory to use for temporary files
  -a AOM_ARGS, --aom-args AOM_ARGS
                        arguments to pass to aomenc
  -k, --keep            keep output segments and other temp data even on success
  --scene-threshold SCENE_THRESHOLD
                        scene detection threshold in percent
```
