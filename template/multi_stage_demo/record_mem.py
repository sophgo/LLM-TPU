import time
import argparse
import json
import os
import traceback

parser = argparse.ArgumentParser('OPT generation script', add_help=False)
parser.add_argument('--logname', type = str, required=True)
args = parser.parse_args()

log_dir = "./log"
os.makedirs(log_dir, exist_ok=True)

def execCmd(cmd):
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return int(text)

max_memory = 0

while True:
    try:
        cmd = "ps -ux | grep pipeline.py | grep -v grep | awk '{ print $6 }'"
        memory = execCmd(cmd)
        if max_memory < memory:
            max_memory = memory
        print("当前内存占用大小: {}K，最大内存占用大小: {}K".format(memory, max_memory))
        time.sleep(1)
        res = {"rss(KB)": memory, "max_rss(KB)": max_memory}
        with open(os.path.join(log_dir, args.logname), "a+", encoding="utf-8") as fp:
            fp.write("{}\n".format(json.dumps(res)))
    except:
        print(traceback.format_exc())
        exit()

