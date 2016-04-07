#!/usr/bin/env python
# encoding: utf-8
import re
def paser(text):
    brackets =  re.findall("(?<=[(])[^()]+\.[^()]+(?=[)])",text)
    return brackets


v = paser("skdf23斤(sdfr)")
print v