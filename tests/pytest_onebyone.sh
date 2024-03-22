echo "Test results" > allres.xml
for ff in ./test_*.py ; do
    echo "Running $ff"
    pytest $ff -v --junitxml=one.xml
    cat one.xml >> allres.xml
    echo "\n" >> allres.xml
done
