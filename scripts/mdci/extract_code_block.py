import re
import markdown
from collections import OrderedDict

__all__ = ["get_all_python_blocks", "get_all_text_blocks", "pickup_blocks"]

def get_markdown_blocks(filepath, pattern, strict = True):
    codeblocks = []
    codeblock_re = r"^```.*"
    codeblock_open_re = pattern+"{0}$".format("" if strict else "?")

    with open(filepath, "r", encoding="utf-8") as f:
        block = []
        python = True
        in_codeblock = False

        for line in f.readlines():
            codeblock_delimiter = re.match(codeblock_re, line)

            if in_codeblock:
                if codeblock_delimiter:
                    if python:
                        codeblocks.append("".join(block))
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

def get_all_python_blocks(filepath, strict=True):
    return get_markdown_blocks(filepath, r"^```(`*)(py|python)", strict)

def get_all_text_blocks(filepath, strict=True):
    return get_markdown_blocks(filepath, r"^```(`*)(text)", strict)


def pickup_blocks(all_blocks, index):
    sub_blocks = OrderedDict()
    
    if isinstance(index, list):
        for i in index:
            sub_blocks[i] = all_blocks[i]
        return sub_blocks
    elif isinstance(index, str):
        if index != 'all':
            raise RuntimeError("index can only be 'all' if it is str")
        index = [x for x in range(0, len(all_blocks))]
        return OrderedDict(zip(index, all_blocks))
    else:
        raise RuntimeError("index can be list only or literal string - 'all'")

if __name__ == "__main__":
    codes = get_all_python_blocks("../../en/docs/basics/02_tensor.md")
    #print(codes)
    picked_codes = pickup_blocks(codes, 'all')
    #print(picked_codes)
    #run_block_item(picked_codes)
    
    run_block_item(pickup_blocks(codes, [1, 5, 6]))
    #texts = get_all_text_blocks("../../en/docs/basics/02_tensor.md")
    #print(texts)