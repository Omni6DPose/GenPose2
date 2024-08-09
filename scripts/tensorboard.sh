#!/bin/bash
tensorboard --logdir ./results/logs/ --port 18090 --reload_interval 1 --samples_per_plugin images=999
