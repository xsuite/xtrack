#!/bin/bash

test -d res_onmom || mkdir res_onmom
cp fort.2 fort.8 fort.16 res_onmom
cp fort.3.onmom res_onmom/fort.3
(cd res_onmom; sixtrack >fort.6)

test -d res_offmom || mkdir res_offmom
cp fort.2 fort.8 fort.16 res_offmom
cp fort.3.offmom res_offmom/fort.3
(cd res_offmom; sixtrack >fort.6)
