import enum
import os
import re
import glob
import copy
# from signal import raise_signal
import traceback
from os.path import join
from io import StringIO
import contextlib
import sys
import yaml

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

def get_textblocks(filepath, safe):
    codeblocks = []
    codeblock_re = r'^```.*'
    codeblock_open_re = r'^```(`*)(text){0}$'.format('' if safe else '?')

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


    current_path = os.path.dirname(os.path.abspath(__file__))
    config_name = 'docs.yml'
    config_file = join(current_path, config_name)


    with open(config_file) as f:
        config = yaml.load(f,Loader=yaml.Loader) # 加载

        path = os.path.abspath(os.path.join(os.getcwd(), "..")) # ../oneflow-documentation/
        file_list = []

        for root, dirs, files in os.walk(path): #遍历仓库中所有的 md 文件，以列表形式写入 file_list
            for file in files:
                if os.path.splitext(file)[1] == '.md':
                    # print(join(root, file))
                    file_list.append(join(root, file))

        '''file_list_check = copy.deepcopy(file_list) # 深拷贝一份 file_list 用于下一步检测

        for section in config: # 遍历一遍，看看有无遗漏的 md 文件
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

        for key, args in config.items(): # 正式开始操作
            filepath = join(path, key)
            run_list = []
            test_list = []
            runAll = False
            testAll = False

            if args:
                if 'RUN' in args:
                    if isinstance(args['RUN'], str):
                        if args['RUN'].strip() == 'all' or args['RUN'].strip() == 'All' or args['RUN'].strip() == 'ALL':
                            runAll = True
                    else:
                        run_list = args['RUN']
                if 'TEST' in args:
                    if isinstance(args['TEST'], str):
                        if args['TEST'].strip() == 'all' or args['TEST'].strip() == 'All' or args['TEST'].strip() == 'ALL':
                            testAll = True
                    else:
                        test_list = args['TEST']


            test_list = list(map(int, test_list))
            run_list = list(map(int, run_list))



            

            # 读取文件
            codeblocks = collect_codeblocks(filepath, safe)
            textblocks = get_textblocks(filepath,safe)

            # 检测是否为负数

            for i in test_list:
                if i < 0:
                    i+=len(textblocks)
            
            for i in run_list:
                if i < 0:
                    i+=len(codeblocks)
                

            if codeblocks:
                singleblock = ''

                for i, blockitem in enumerate(codeblocks):
                    if i in run_list or runAll:
                        singleblock += blockitem
                        if not runAll:
                            run_list.remove(i)

                if run_list != []: raise ValueError("The RUN args for {} has indexes that does not exist.".format(filepath))
                
                
                
                @contextlib.contextmanager
                def stdoutIO(stdout=None):
                    old = sys.stdout
                    if stdout is None:
                        stdout = StringIO()
                    sys.stdout = stdout
                    yield stdout
                    sys.stdout = old
                print("Running " + filepath)
                with stdoutIO() as s:
                    try:
                        
                        exec(singleblock, globals())
                    except Exception as e:
                        traceback.print_exc()
                print(s.getvalue())
                for i, blockitem in enumerate(textblocks):
                    if i in test_list or testAll:
                        if i not in run_list and not runAll:
                            raise ValueError("The TEST args contains indexes that does not exist in RUN args.")
                        if blockitem not in s.getvalue():
                            raise ValueError("The text block:\n {} does not match the code output.".format(blockitem))
                        test_list.remove(i)
                if test_list != [] : raise ValueError("The TEST config for {} has indexes that does not exist.".format(filepath))
                
                print("ok")


if __name__ == "__main__":
    main()
    