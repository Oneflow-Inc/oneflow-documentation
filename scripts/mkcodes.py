import os
import re
import glob
import copy
import traceback
from configparser import ConfigParser
from os.path import join

try:
    import markdown as markdown_enabled
except ImportError:
    markdown_enabled = False
else:
    from markdown.extensions import Extension
    from markdown.treeprocessors import Treeprocessor


def github_codeblocks(filepath, safe):
    codeblocks = []
    codeblock_re = r'^```.*'
    codeblock_open_re = r'^```(`*)(py|python){0}$'.format('' if safe else '?')

    with open(filepath, 'r', encoding="utf-8") as f:
        block = []
        python = True
        in_codeblock = False

        for line in f.readlines():
            codeblock_delimiter = re.match(codeblock_re, line)

            if in_codeblock:
                if codeblock_delimiter:
                    if python:
                        codeblocks.append(''.join(block))
                    block = []
                    python = True
                    in_codeblock = False
                else:
                    block.append(line)
            elif codeblock_delimiter:
                in_codeblock = True
                if not re.match(codeblock_open_re, line):
                    python = False
    return codeblocks


def is_markdown(f):
    markdown_extensions = ['.markdown', '.mdown', '.mkdn', '.mkd', '.md']
    return os.path.splitext(f)[1] in markdown_extensions


def get_nested_files(directory, depth):
    for i in glob.iglob(directory + '/*'):
        if os.path.isdir(i):
            yield from get_nested_files(i, depth+1)
        elif is_markdown(i):
            yield (i, depth)


def get_files(inputs):
    for i in inputs:
        if os.path.isdir(i):
            yield from get_nested_files(i, 0)
        elif is_markdown(i):
            yield (i, 0)


def makedirs(directory):
    to_make = []

    while directory:
        try:
            os.mkdir(directory)
        except FileNotFoundError:
            directory, tail = os.path.split(directory)
            to_make.append(tail)
        else:
            if to_make:
                directory = os.path.join(directory, to_make.pop())
            else:
                break


def main():
    collect_codeblocks = github_codeblocks
    safe = True

    config = ConfigParser()

    current_path = os.path.dirname(os.path.abspath(__file__))
    config_name = 'docs.yml'
    config_file = join(current_path, config_name)

    config.read(config_file)
    section_list = config.sections() # 捕获所有 section

    path = os.path.abspath(os.path.join(os.getcwd(), "..")) # ../oneflow-documentation/
    print(path)
    file_list = []

    for root, dirs, files in os.walk(path): #遍历仓库中所有的 md 文件，以列表形式写入 file_list
        for file in files:
            if os.path.splitext(file)[1] == '.md':
                # print(join(root, file))
                file_list.append(join(root, file))

    '''file_list_check = copy.deepcopy(file_list) # 深拷贝一份 file_list 用于下一步检测

    for section in section_list: # 遍历一遍，看看有无遗漏的 md 文件
        filepath = join(path, section)
            
        if os.path.exists(filepath) and filepath in file_list_check:
            print(filepath+" exists.")
            file_list_check.remove(filepath)
        else:
            raise ValueError(
                'The file path {} does not exist.'.format(filepath))
    

    if file_list_check != []:
        raise ValueError('The following files are not recorded in {}. \n {}'.format(config_name, file_list_check))
    '''

    for section in section_list: # 正式开始操作
        filepath = join(path, section)
        run_list = []
        test_list = []
        runAll = False

        if 'run' in config.options(section): # 读取 configs
            
            if config.get(section, 'run').strip() == 'all' or config.get(section, 'run').strip() == 'All':
                runAll = True
            else:
                run_list = config.get(section, 'run').replace(
                    ' ', '').split(",")
                for i in run_list:
                    print(i)

        if 'test' in config.options(section):
            test_list = config.get(section, 'test').replace(
                    ' ', '').split(",")


         # 读取文件
        codeblocks = collect_codeblocks(filepath, safe)

        if codeblocks:
            singleblock = ''

            for i, blockitem in enumerate(codeblocks):
                if i in run_list or runAll:
                    singleblock += blockitem
                    if not runAll:
                        run_list.remove(i)
                if str(i) in test_list:
                    if str(i) not in run_list and not runAll:
                        raise ValueError("All the test block should be included in run block in {}.".format(filepath))
                    else:
                        try:
                            print("testing block {} of file {}".format(i,filepath))
                        except Exception as e:
                            print("Test failed for block {} in file {}".format(i,filepath))
                        else:
                            print("test passed")
                        test_list.remove(str(i))
            print(test_list)
            # if run_list != []: raise ValueError("The run config for {} has indexes that does not exist.".format(filepath))
            # if test_list != [] : raise ValueError("The test config for {} has indexes that does not exist.".format(filepath))
            
            try:
                print("Running " + filepath)
                exec(singleblock, globals())
            except Exception as e:
                traceback.print_exc()
            else:
                print("ok")
    












if __name__ == "__main__":
    main()
    