
`nvcli` 提供了两个字命令，`mount` 和 `unmount` 子命令；通过设置 `—v` 可以查看更详细的执行日志。

```bash
 $ ./nvcli
Support GPU hot loading and unloading to a Pod command line tool

Usage:
  nvcli [command]

Available Commands:
  help        Help about any command
  mount       dynamic mount nvidia gpu device to pod
  unmount     dynamic unmount nvidia gpu device to pod
  version     Print the version number of Hugo

Flags:
  -h, --help                help for nvcli
      --kubeconfig string   kubeconfig file (default "/root/.kube/config")
      --name string         pod name
      --namespace string    namespace (default "default")
      --v string            Log level for klog (0-10) (default "2")

Use "nvcli [command] --help" for more information about a command.
```

### mount

mount 提供了一个 `—-mount` 的参数，主要是指定宿主机上 GPU 的 index，然后会被加载到容器中，可以接受一个数组。

```bash
$ ./nvcli mount
Usage:
  nvcli mount [flags]

Flags:
  -h, --help                help for mount
      --mount stringArray   mount nvidia gpu device index

Global Flags:
      --kubeconfig string   kubeconfig file (default "/root/.kube/config")
      --name string         pod name
      --namespace string    namespace (default "default")
      --v string            Log level for klog (0-10) (default "2")
```

如下是一个参数 demo，指定 pod 的name，pod namespace 默认是defaule, `--kubeconfig` 默认读区当前根目录下的文件。

```bash
$ ./nvcli mount --name=gpu-pod --mount=0
```

### unmount

`unmount` 提供了一个 `—-mount` 的参数，主要是指定宿主机上 GPU 的 index，然后会被从容器中卸载，可以接受一个数组。

```bash
$ ./nvcli unmount
Usage:
  nvcli unmount [flags]

Flags:
  -h, --help                  help for unmount
      --unmount stringArray   unmount nvidia gpu device index

Global Flags:
      --kubeconfig string   kubeconfig file (default "/root/.kube/config")
      --name string         pod name
      --namespace string    namespace (default "default")
      --v string            Log level for klog (0-10) (default "2")

```

如下是一个参数 demo，指定 pod 的name，pod namespace 默认是defaule, `--kubeconfig` 默认读区当前根目录下的文件。

更详细的原理请看: [GPU 设备动态挂载到 Pod 原理分析](https://mp.weixin.qq.com/s/bYIsNlVkOZDjT2GmBltiTg)
