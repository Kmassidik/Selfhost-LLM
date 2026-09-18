#!/bin/bash
awk -F", " "{m[\$1]=(\$2+0>m[\$1])?\$2+0:m[\$1]; u[\$1]=(\$3+0>u[\$1])?\$3+0:u[\$1]; s[\$1]+=\$3; n[\$1]++; if(\$3+0>5)busy[\$1]++} END{for(k in m) printf \"card %s: peak %5d MiB | max util %3d%% | mean util %5.1f%% | samples>5%% util: %d/%d\n\", k, m[k], u[k], s[k]/n[k], busy[k]+0, n[k]}" "$1" | sort
