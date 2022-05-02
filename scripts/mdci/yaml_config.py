import yaml

def read_config(yaml_file):
    with open(yaml_file) as f:
        config = yaml.load(f, Loader=yaml.Loader)
        return config

if __name__ == "__main__":
    cfg01 = read_config("./configs/cn_docs_basics_08_graph.yml")
    cfg02 = read_config("./configs/en_docs_basics_02_tensor.yml")
    print(cfg01)
    print(cfg02)