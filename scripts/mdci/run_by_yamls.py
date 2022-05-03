import yaml
import subprocess
import os
import argparse

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
CONFIG_DIR = os.path.join(BASE_DIR, "scripts/mdci/configs")
CN_DOCS = os.path.join(BASE_DIR, "cn/docs")
EN_DOCS = os.path.join(BASE_DIR, "en/docs")

def read_config(yaml_file):
    with open(yaml_file) as f:
        config = yaml.load(f, Loader=yaml.Loader)
        return config


def run_yaml_markdown_codes(yaml_path, config, all_markdown_files):
    file_path = os.path.join(BASE_DIR, config["file_path"])
    if all_markdown_files:
        try:
            all_markdown_files.remove(file_path)
        except:
            pass  # do nothing if remove more than once
    for index in config["run"]:
        cmd = r"python3 run_markdown_codes.py --markdown_file {0} --index {1}".format(
            file_path, str(index).replace(" ", "")
        )
        cmd_list = cmd.split(" ", 5)
        print("====RUN====:", cmd)
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


def run_all_yamls(all_markdown_files):
    print(get_all_yaml_files())
    for yaml_file in get_all_yaml_files():
        run_configs_in_yaml(yaml_file, all_markdown_files)
    print("MARKDOWN FILES NOT TEST:")
    print("\n".join(all_markdown_files))

def run_configs_in_yaml(yaml_file, all_markdown_files=None):
    for config in read_config(yaml_file):
        run_yaml_markdown_codes(yaml_file, config, all_markdown_files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="read config yaml files and run realted code"
    )
    parser.add_argument(
        "--yaml", type=str, default=None, help="the path of yaml file. eg: ./sample.yaml"
    )
    args = parser.parse_args()

    if args.yaml:
        run_configs_in_yaml(args.yaml)
    else:
        markdown_files = get_all_markdown_files()
        run_all_yamls(markdown_files)

