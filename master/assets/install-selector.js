; (function () {
    window.addEventListener('load', () => {

        function get_commands(latest_version) {
            let stable_command_116 = 'python3 -m pip install -f https://release.oneflow.info oneflow==VERSION+cu116'
            let stable_command_117 = 'python3 -m pip install -f https://release.oneflow.info oneflow==VERSION+cu117'
            let stable_command_cpu = 'python3 -m pip install -f https://release.oneflow.info oneflow==VERSION+cpu'
            let commands = [
                {
                    versions: 'Stable',
                    framework: 'CUDA',
                    smlVers: '11.6',
                    command: stable_command_116.replace("VERSION", latest_version)
                },
                {
                    versions: 'Stable',
                    framework: 'CUDA',
                    smlVers: '11.7',
                    command: stable_command_117.replace("VERSION", latest_version)
                },
                {
                    versions: 'Stable',
                    framework: 'CPU',
                    smlVers: '',
                    command: stable_command_cpu.replace("VERSION", latest_version)
                },
                {
                    versions: 'Nightly',
                    framework: 'CUDA',
                    smlVers: '11.6',
                    command: 'python3 -m pip install -f https://staging.oneflow.info/branch/master/cu116 --pre oneflow'
                },
                {
                    versions: 'Nightly',
                    framework: 'CUDA',
                    smlVers: '11.7',
                    command: 'python3 -m pip install -f https://staging.oneflow.info/branch/master/cu117 --pre oneflow'
                },
                {
                    versions: 'Nightly',
                    framework: 'CPU',
                    smlVers: '',
                    command: 'python3 -m pip install -f https://staging.oneflow.info/branch/master/cpu --pre oneflow'
                },
            ]
            return commands
        }

        function init_selector(commands) {
            let condition = {
                versions: 'Stable',
                framework: 'CUDA',
                smlVers: '11.6',
            }
            selectCommands(condition)
            let items = document.querySelectorAll('#instruction li')

            function selectCommands(conditioning) {
                let filter = null
                if (conditioning.framework == "CPU") {
                    filter = commands.filter(e => e.versions == conditioning.versions).filter(e => e.framework == conditioning.framework)
                } else {
                    filter = commands.filter(e => e.versions == conditioning.versions).filter(e => e.framework == conditioning.framework).filter(e => e.smlVers == conditioning.smlVers)
                }
                if (filter && filter[0]) {
                    document.querySelector('.panel-code').innerHTML = filter[0].command
                }
            }
            items.forEach(e => {
                e.addEventListener('click', function () {
                    let attach = this.getAttribute('attach')
                    let tempItems = document.querySelectorAll(`[attach=${attach}]`)
                    tempItems.forEach(e => {
                        e.className = ''
                    })
                    this.className = 'active'
                    condition[attach] = this.innerHTML
                    if (condition['framework'] == 'CPU') {
                        document.querySelector('.smlVers').style.height = '0px'
                    } else {
                        document.querySelector('.smlVers').style.height = '48px'
                    }
                    selectCommands(condition)
                })
            })
        }

        let TAGS_API_URL = 'https://api.github.com/repos/Oneflow-Inc/oneflow/tags'
        let xmlhttp = new XMLHttpRequest();
        let latest_version_hardcode = "0.8.0" // using latest version in hard-code way if request fails
        xmlhttp.onreadystatechange = function () {
            if (xmlhttp.readyState == 4) {// 4 = "loaded"
                if (xmlhttp.status == 200) {// 200 = "OK"
                    localStorage.latest_version = eval(xmlhttp.responseText)[0].name.replace("v", "").replace("0.8.1", "0.8.0") // eg: v0.x.0 => 0.x.0
                    init_selector(get_commands(localStorage.latest_version))
                }
                else {
                    init_selector(get_commands(localStorage.latest_version ? localStorage.latest_version : latest_version_hardcode))
                }
            }
        }
        xmlhttp.open("GET", TAGS_API_URL, true)
        xmlhttp.send(null)
    })
})();

function copyPipCommand() {
    var copyText = document.querySelector('.panel-code').innerHTML
    navigator.clipboard.writeText(copyText)
}
