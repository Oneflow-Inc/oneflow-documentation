
## 框架设计

## 源码目录结构

## 修改并提交代码

以[github flow](https://guides.github.com/introduction/flow/)为例。

### 克隆仓库并更新代码
通过以下命令，克隆oneflow远程仓库：
```shell
git clone git@github.com:Oneflow-Inc/oneflow.git
```

切换到`develop`分支：
```shell
git checkout develop
```

同步最新代码：
```shell
git pull origin develop
```


### 创建本地分支

所有的新增功能或者bug修复，都应该在本地的新分支完成，通过以下命令创建新分支：
```shell
git checkout -b my-features
```

在此基础上，你可以对oneflow源码进行开发。开发完成后，执行提交（`commit`）操作。


### commit & push

通过`git commit`可以提交你的修改。【我们有触发CI的操作吗】
```shell
git commit -m "comment on commit"
```

commit之后，通过`git push`命令将更新同步到远端仓库:
```shell
git push origin my-features
```

## 创建提交PR
