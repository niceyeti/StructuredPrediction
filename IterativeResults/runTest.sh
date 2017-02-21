echo "Running ocr datasets, for maxIt=1"
python ../StructuredPerceptron.py --trainPath=../ocr_fold0_sm_train.txt --testPath=../ocr_fold0_sm_test.txt --maxIt=1 --R=20 --eta=0.01 --phi=1 > ocrIts.txt
echo "Running ocr datasets, for maxIt=10"
python ../StructuredPerceptron.py --trainPath=../ocr_fold0_sm_train.txt --testPath=../ocr_fold0_sm_test.txt --maxIt=10 --R=20 --eta=0.01 --phi=1 >> ocrIts.txt
echo "Running ocr datasets, for maxIt=25"
python ../StructuredPerceptron.py --trainPath=../ocr_fold0_sm_train.txt --testPath=../ocr_fold0_sm_test.txt --maxIt=25 --R=20 --eta=0.01 --phi=1 >> ocrIts.txt
echo "Running ocr datasets, for maxIt=50"
python ../StructuredPerceptron.py --trainPath=../ocr_fold0_sm_train.txt --testPath=../ocr_fold0_sm_test.txt --maxIt=50 --R=20 --eta=0.01 --phi=1 >> ocrIts.txt
echo "Running ocr datasets, for maxIt=100"
python ../StructuredPerceptron.py --trainPath=../ocr_fold0_sm_train.txt --testPath=../ocr_fold0_sm_test.txt --maxIt=100 --R=20 --eta=0.01 --phi=1 >> ocrIts.txt


echo "Running net datasets, for maxIt=1"
python ../StructuredPerceptron.py --trainPath=../nettalk_stress_train.txt --testPath=../nettalk_stress_test.txt --maxIt=1 --R=20 --eta=0.01 --phi=1 > netIts.txt
echo "Running net datasets, for maxIt=10"
python ../StructuredPerceptron.py --trainPath=../nettalk_stress_train.txt --testPath=../nettalk_stress_test.txt --maxIt=10 --R=20 --eta=0.01 --phi=1 >> netIts.txt
echo "Running net datasets, for maxIt=25"
python ../StructuredPerceptron.py --trainPath=../nettalk_stress_train.txt --testPath=../nettalk_stress_test.txt --maxIt=25 --R=20 --eta=0.01 --phi=1 >> netIts.txt
echo "Running net datasets, for maxIt=50"
python ../StructuredPerceptron.py --trainPath=../nettalk_stress_train.txt --testPath=../nettalk_stress_test.txt --maxIt=50 --R=20 --eta=0.01 --phi=1 >> netIts.txt
echo "Running net datasets, for maxIt=100"
python ../StructuredPerceptron.py --trainPath=../nettalk_stress_train.txt --testPath=../nettalk_stress_test.txt --maxIt=100 --R=20 --eta=0.01 --phi=1 >> netIts.txt

