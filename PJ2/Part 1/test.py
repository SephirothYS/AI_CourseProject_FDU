import HMM
import check

E_train_data = "./Part 1/English/train.txt"
E_test_data = "./Part 1/English/validation.txt"
E_output_path = "./Part 1/Output/output(E).txt"
C_train_data = "./Part 1/Chinese/train.txt"
C_test_data = "./Part 1/Chinese/validation.txt"
C_output_path = "./Part 1/Output/output(C).txt"

# E_test_data = "H:/english_test.txt"
# C_test_data = "H:/chinese_test.txt"

model = HMM.HMM()

mode = "English"
if mode == "English":
    model.train(E_train_data)
    model.predict(E_test_data,E_output_path)
    check.check("English",E_test_data,E_output_path)
else:
    model.train(C_train_data)
    model.predict(C_test_data,C_output_path)
    check.check("Chinese",C_test_data,C_output_path)