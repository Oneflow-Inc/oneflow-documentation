;(function(){
    window.addEventListener('load', () => {
        function get_commands(){
            let commands = [
                {
                    versions: 'Stable',
                    framework: 'CUDA',
                    smlVers: '10.2',
                    command: 'python3 -m pip install -f https://release.oneflow.info oneflow==0.6.0+cu102'
                },
                {
                    versions: 'Stable',
                    framework: 'CUDA',
                    smlVers: '11.2',
                    command: 'python3 -m pip install -f https://release.oneflow.info oneflow==0.6.0+cu112'
                },
                {
                    versions: 'Nightly',
                    framework: 'CUDA',
                    smlVers: '10.2',
                    command: 'python3 -m pip install -f https://staging.oneflow.info/branch/master/cu102 --pre oneflow'
                },
                {
                    versions: 'Nightly',
                    framework: 'CUDA',
                    smlVers: '11.2',
                    command: 'python3 -m pip install -f https://staging.oneflow.info/branch/master/cu112 --pre oneflow'
                },
                {
                    versions: 'Stable',
                    framework: 'CPU',
                    smlVers: '',
                    command: 'python3 -m pip install -f https://release.oneflow.info oneflow==0.6.0+cpu'
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
        let commands = get_commands()
        let condition = {
        versions: 'Stable',
        framework: 'CUDA',
        smlVers: '10.2',
        }
        selectCommands(condition)
        let items = document.querySelectorAll('#instruction li')
        
        function selectCommands(conditioning) {
            let filter = null
        if(conditioning.smlVers == "CPU"){
            filter = commands.filter(e => e.versions == conditioning.versions).filter(e => e.framework == conditioning.framework)
        }else{
            filter = commands.filter(e => e.versions == conditioning.versions).filter(e => e.framework == conditioning.framework).filter(e => e.smlVers == conditioning.smlVers)
        }
        if (filter && filter[0]) {
            document.querySelector('.panel-code').innerHTML = filter[0].command
        }
        }
        items.forEach(e => {
        e.addEventListener('click', function() {
            let attach = this.getAttribute('attach')
            let tempItems = document.querySelectorAll(`[attach=${attach}]`)
            tempItems.forEach(e => {
            e.className = ''
            })
            this.className = 'active'
            condition[attach] = this.innerHTML
            if (this.innerHTML == 'CPU') {
            condition['smlVers'] = 'CPU'
            document.querySelector('.smlVers').style.height = '0px'
            } else {
            document.querySelector('.smlVers').style.height = '48px'
            }
            console.log(condition)
            selectCommands(condition)
        })
        })
    })
    })();
