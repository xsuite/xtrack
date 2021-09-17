#!/bin/bash

test -d res || mkdir res
cp fort.2 fort.3 fort.8 fort.16 res
(cd res; sixtrack >fort.6)
