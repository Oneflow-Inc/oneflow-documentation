文档采用markdown编写，采用`mkdocs`文档生成工具及`material`主题。

**安装mkdocs与material**

```shell
pip3 install mkdocs mkdocs-material --user
```

**编译为html**

```shell
mkdocs build
```

编译结果保存在`site`目录下，文档属性设置见`mkdocs.yml`。

**其它**

mkdocs-material主题使用了google字体资源，可能导致加载缓慢。
可以找到主题目录下的`base.html`，将相关代码注释掉即可:
```xml
<!--
<link href="https://fonts.gstatic.com" rel="preconnect" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css?family={{
	font.text | replace(' ', '+') + ':300,400,400i,700%7C' +
	font.code | replace(' ', '+')
  }}&display=fallback">
		  -->
```
