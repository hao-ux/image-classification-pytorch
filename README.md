下面为一个shell脚本程序。运行后发现结果不对。请使用set命令打开-x选项调试该程序。

#!/bin/bash
for((x=0;x<=20;++x))
do
    for((y=0;y<34;++y))
    do
        ((z=100-x-y))
        ((v=(z%3==0)&&(5*3+3*y+z/3==100)))
        if ((v&&(x&&y&&z)))
        then
            echo "cock=$x***hen=$y***chicken=$z"
            echo "This is one of solutions."
        fi
    done
    break
done
exit