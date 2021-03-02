CFG='weights/tosnet_ours/config.txt'
WEIGHTS='weights/tosnet_ours/models/TOSNet_epoch-49.pth'
TEST=false
EVAL=true

if $TEST
then
    python test.py --test_set coift --cfg $CFG --weights $WEIGHTS
    python test.py --test_set hrsod --cfg $CFG --weights $WEIGHTS
    python test.py --test_set thinobject5k_test --cfg $CFG --weights $WEIGHTS
fi

if $EVAL
then
    python eval.py --test_set coift --result_dir 'results/coift'
    python eval.py --test_set hrsod --result_dir 'results/hrsod'
    python eval.py --test_set thinobject5k_test --result_dir '/results/thinobject5k_test'
fi
