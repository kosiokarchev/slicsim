FILE=emcee

for N in 100000; do
  for datatype in photoz; do
    for suffix in 0; do
      for versions in "0 1 2" "3 4 5" "6 7" "8 9"; do
        for version in `echo $versions`; do
          nohup python scripts/run_$FILE.py --N=$N --datatype=$datatype --suffix=$suffix --version=$version > nohup/$FILE-$N-$datatype-$suffix-$version.out &
        done
        wait
      done
    done
  done
done