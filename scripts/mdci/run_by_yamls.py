import yaml
import subprocess
import os
import argparse

parser = argparse.ArgumentParser(
    description="read config yaml files and run realted code"
)
parser.add_argument(
    "--yaml", type=str, default=None, help="the path of yaml file. eg: ./sample.yaml"
)
args = parser.parse_args()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
CONFIG_DIR = os.path.join(BASE_DIR, "scripts/mdci/configs")
CN_DOCS = os.path.join(BASE_DIR, "cn/docs")
EN_DOCS = os.path.join(BASE_DIR, "en/docs")


def read_config(yaml_file):
    with open(yaml_file) as f:
        config = yaml.load(f, Loader=yaml.Loader)
        return config


def run_yaml_markdown_codes(yaml_path, config):
    file_path = os.path.join(BASE_DIR, config["file_path"])
    try:
        ALL_MARKDOWN_FILES.remove(file_path)
    except:
        pass  # do nothing if remove more than once
    for index in config["run"]:
        cmd = r"python3 run_markdown_codes.py --markdown_file {0} --index {1}".format(
            file_path, str(index).replace(" ", "")
        )
        cmd_list = cmd.split(" ", 5)
        subprocess_ret = subprocess.run(cmd_list)
        if subprocess_ret.returncode != 0:
            print("ERROR!!! YAML {0} fails when run: {1}".format(yaml_path, cmd_list))


def get_all_yaml_files():
    yaml_files_list = []
    for root, dirs, files in os.walk(CONFIG_DIR):
        for file in files:
            if os.path.splitext(file)[1] == ".yml":
                yaml_files_list.append(os.path.join(root, file))
    return yaml_files_list


def get_all_markdown_files():
    md_files_list = []
    for root, dirs, files in os.walk(CN_DOCS):
        for file in files:
            if os.path.splitext(file)[1] == ".md":
                md_files_list.append(os.path.join(root, file))
    for root, dirs, files in os.walk(EN_DOCS):
        for file in files:
            if os.path.splitext(file)[1] == ".md":
                md_files_list.append(os.path.join(root, file))
    return md_files_list


ALL_MARKDOWN_FILES = get_all_markdown_files()


def main():
    for yaml_file in get_all_yaml_files():
        run_yaml_markdown_codes(yaml_file, read_config(yaml_file))


if __name__ == "__main__":
    if args.yaml:
        run_yaml_markdown_codes(args.yaml, read_config(args.yaml))
    else:
        main()
        print("MARKDOWN FILES NOT TEST:")
        print("\n".join(ALL_MARKDOWN_FILES))
