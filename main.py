import numpy as np
import row as row
from scipy.optimize import linprog
import matplotlib.pyplot as plt

# in các kí tự sử dụng vòng lặp for
def print_hi(name):
    for i in name:
        print(i)

# in số lẻ trong phạm vi từ 1 1<= x <=10
def oddNumber():
    sum =0;
    for i in range (1,10):
        if i % 2 != 0:
            sum += i;
            print(i, end="  ")
    #3A
    print("\ntong cac so le:", sum)

#3B. tÍNH tổng các số từ 1 đến 6
def Sum():
    sum = 0;
    for i in range(1,6):
        sum += i;
    print("tổng các số từ 1 đến 6 là : ", sum)
#4/ Given mydict={“a”: 1,”b”:2,”c”:3,”d”:4}.
def myDict():
    mydict = {"a": 1,"b":2,"c":3,"d":4}
    #a/ Print all key in mydict
    for i in mydict.keys():
        print(i,end="  ")
    #b/ Print all values in mydict
    for i in mydict.values():
        print(i, end="  ")
    #c/ Print all keys and values
    for i in mydict.items():
        print(i, end="  ")
#5/ Given courses=[131,141,142,212] and names=[“Maths”,”Physics”,”Chem”, “Bio”].
# Print a sequence of tuples, each of them contains one courses and one names
def courses():
    courses = [131, 141, 142, 212];
    names = ["Maths", "Physics", "Chem", "Bio"]
    zipped = zip(courses, names)
    print(list(zipped))

#6/ Find the number of consonants in “jabbawocky” by two ways ( tìm phụ âm)
	#a/ Directly (i.e without using the command “continue”)
def Consonants():
    word = "jabbawocky"
    count = 0;
    for i in word:
        if i.lower() not in "ueoai":
            count +=1;
    print("số phụ âm là: ", count)

	#b/ Check whether it’s characters are in vowels set and using the command “continue
    count2 = 0;
    for i in word:
        if i.lower() in "ueoai":
            continue;
        count2 += 1;
    print("số phụ âm là: ", count2)

#7/ a is a number such that -2<=a<3. Print out all the results of 10/a using try…except.
# When a=0, print out “can’t divided by zero”
def dividedNumber():
    for a in range(-2,3):
        try:
            result = 10 / a
            print("10 /", a, "=",result)
        except ZeroDivisionError:
            print("Không thể chia cho 0")

#8/ Given ages=[23,10,80]
#And names=[Hoa,Lam,Nam]. Using lambda function to sort
# a list containing tuples (“age”,”name”) by increasing of the ages
def lambdaSortList():
    names = ["Hoa","Lam", "Nam"]
    ages = [23, 10, 80]

    DanhSach = list(zip(ages, names))
    print("list:" , DanhSach)

    DSSapXep = sorted(DanhSach, key= lambda x :x[0])
    print("Dang sách sx theo tuổi:", DSSapXep)

#9/ Create  a file “firstname.txt”:
#a/ Open this file for reading
#b/Print each line of this file
#c/ Using .read to read the file and
def UseFile():
    input_file = open("data.txt")
    for line in input_file:
        print(line, end='')
    #read file
    input_file = open("data.txt")
    data = input_file.readlines()
    input_file.close()
    print(data)


#/ Define a function that return the sum of two numbers a and b. Try with a=3, b=4.
def SumTwoNumber(a, b):
    return a+b;

#Create a 3x3 matrix
#And check the rank and the shape of this matrix and vector v.
def matrix():
    matran = np.array([[1, 2,3], [4,5,6], [7,8,9]])
    vector = np.array([1,2,3])

    rank = np.linalg.matrix_rank(matran)
    result = np.linalg.matrix_rank(rank)
    print("Ma trận:", matran)
    print("Thứ hạng của ma trận:", rank)
    print("Hình dạng của ma trận:", matran.shape)
    print("Hình dạng của vector:", vector.shape)

#/3 Create a new 3x3 matrix such that its’ elements are
# the sum of corresponding (position) element of M plus 3
def newMatrix():
    matran = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    matran_moi = matran + 3 * np.arange(1,10).reshape(3,3)
    print(matran_moi)

    #4 / Create the transpose of M and v
    vector = np.array([1, 2, 3])
    # chuyển vị
    vector_chuyenvi = np.transpose(vector)
    matran_chuyenvi = np.transpose(matran_moi)
    print("vector:" , vector)
    print("vector chuyển vị : ", vector_chuyenvi)
    print("ma tran chuyển vị: ", matran_chuyenvi)

#Compute the norm of x=(2,7). Normalization vector x.
def Norm():
    x = np.array([2,7])
    #tính định mức
    norm_x = np.linalg.norm(x)

    print (" Định mức của vector x: ", norm_x)
    #vector chuẩn hóa
    print("vector chuẩn hóa:" , x /norm_x)

#6 Given a=[10,15], b=[8,2] and c=[1,2,3]. Compute a+b, a-b, a-c. Do all of them work? Why?
def Compute():
    a = np.array([10,15])
    b = np.array([8, 2])
    c = np.array([1,2,3])
    print("cộng 2 ma trận a và b : ", a + b)
    print("trừ 2 ma trận a b: ", a - b)
    try:
        print("trừ hai ma trận a- c : ", a - c)
    except ValueError as e:
        print(e)

#7/ Tính tích vô hướng của a và b
def dotMatrix():
    a = np.array([10, 15])
    b = np.array([8, 2])

    print("tích vô hướng của 2 ma trận:", np.dot(a,b))

#8 8/ Given matrix A=[[2,4,9],[3,6,7]].
	#a/ Check the rank and shape of A
	#b/ How can get the value 7 in A?
	#c/ Return the second column of A
def rank_shape_matrix():
    matran = np.array([[2,4,9], [3,6,7]])
    rank = np.linalg.matrix_rank(matran)

    print("hình dạng của ma trân: ", matran.shape)
    print("rank của ma trận:", rank)

    # tìm phần tử 7
    value7_index = np.where(matran == 7)
    print("giá trị 7 trong ma trận là:", matran[value7_index])

    #trả về cột thứ hai
    column2 = matran[:,1]
    print(" cột thứ 2 của ma trận là :", column2)


#9 Create a random  3x3 matrix  with the value in range (-10,10)
def randomMatrix():
    random_matran = np.random.randint(-10,10, size=(3,3))

    print("Ma trận ngâũ nhiên:", random_matran)

#10/ Create an identity (3x3) matrix.( ma trận đơn vị)
def maTranDonVi():
    matran = np.eye(3)
    print(" ma trậ đơn vị:", matran)

#11/ Create a 3x3 random matrix with the value in range (1,10). Compute the trace of this matrix by 2 ways:
	#a/ By one command
	#b/ By using for loops
def matrix11():
    random_matran = np.random.randint(1, 10, size=(3, 3))
    trace_matran = np.trace(random_matran)
    print("ma trận:", random_matran)

    #cách 1
    print("vết của ma trận:", trace_matran)
    #cách 2
    trace_matran2 = 0;
    for i in range(3):
        trace_matran2 += random_matran[i,i];
    print("vết của ma trận dùng vòng lặp:", trace_matran2)

#12/ Create a 3x3 diagonal matrix with the value in main diagonal 1,2,3.
# tạo ma trận với dg chéo  1 2 3
def diagonalMatrix():
    matran = np.diag([1,2,3])
    print(" ma trận với đường chéo chính 1 2 3:", matran)

# 13. Given A=[[1,1,2],[2,4,-3],[3,6,-5]]. Compute the determinant of A
def DinhThucMaTran():
    matran = np.array([[1,1,2], [2,4,-3],[3,6,-5]])

    # tính định thức
    dinhthuc = np.linalg.det(matran)
    print("Ma trận:")
    print(" định thức của ma trận là : ", dinhthuc)

#14 14/ Given a1=[1,-2,-5] and a2=[2,5,6].
# Create a matrix M such that the first column is a1 and the second column is a2
def matrix14():
    a1=[1,-2,-5]
    a2 = [2, 5, 6]

    matran = np.column_stack((a1, a2))
    print(" Ma trận: ", matran)


#15/ Đơn giản chỉ cần vẽ giá trị bình phương của y với y trong phạm vi (-5<=y<6).
def VeBinhPhuong():
    y_values = range(-5,6)

    #tính bình phương
    y_binhphuong = [ y ** 2 for y in y_values]

    # vẽ đồ thị
    plt.plot(y_values, y_binhphuong)

    #đặt tiêu đề và nhãn cho trục x và y
    plt.title("biểu đồ giá trị bình phương của y")
    plt.xlabel("y")
    plt.ylabel('y binh phuong')

    plt.grid(True)
    plt.show()

#16/ Tạo 4 giá trị cách đều nhau từ 0 đến 32 (bao gồm cả điểm cuối)
def bai16():
    values = np.linspace(0,32,num=4)
    print("4 giá trị cách đều nhau từ 0 đến 32:",values)

#17/ Lấy 50 giá trị cách đều nhau từ -5 đến 5 cho x. Tính y=x**2. Đồ thị (x,y).
def bai17():
    x = np.linspace(-5,5, num=50)

    #tính y = x**2
    y = x ** 2
    # vẽ đồ thị
    plt.plot(x,y)

    # đặt tiêu đề và nhãn cho trục x và y
    plt.title("biểu đồ giá trị bình phương của y")
    plt.xlabel("y")
    plt.ylabel('y binh phuong')

    plt.grid(True)
    plt.show()

#18/ Vẽ biểu đồ y=exp(x) có nhãn và tiêu đề.
def bai18():
    x = np.linspace(-5, 5, num=100)

    # tính y = x**2
    y = np.exp((x))
    # vẽ đồ thị
    plt.plot(x, y, label = "y =exp(x)")

    # đặt tiêu đề và nhãn cho trục x và y
    plt.title("biểu đồ y =exp(x)")
    plt.xlabel("x")
    plt.ylabel('y ')

    plt.grid(True)
    plt.show()

#19/ Tương tự với y=log(x) với x từ 0 đến 5.
def bai19():
    x = np.linspace(0.1, 10, num=100)

    y = np.log(x)
    # vẽ đồ thị
    plt.plot(x, y, label="y =log(x)")

    # đặt tiêu đề và nhãn cho trục x và y
    plt.title("biểu đồ y =log(x)")
    plt.xlabel("x")
    plt.ylabel('y ')

    plt.grid(True)
    plt.show()

#20 Vẽ hai đồ thị y=exp(x), y=exp(2*x) trong cùng một đồ thị và y=log(x
#) và y=log(2*x) trong cùng một đồ thị bằng cách sử dụng subplot.

def bai20():
    x = np.linspace(-5, 5, num=100)

    y1 = np.exp(x)
    y2 = np.exp(2 * x)

    y3 = np.log(x)
    y4 =np.log(2 * x)

    # vẽ đồ thị
    plt.subplot(2,1,1)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.title("biểu đồ y =exp(x) và exp(2*x)")
    plt.xlabel("x")
    plt.ylabel('y ')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(x, y3)
    plt.plot(x, y4)
    plt.title("biểu đồ y =log(x) và log(2*x)")
    plt.xlabel("x")
    plt.ylabel('y ')

    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    #1
    print_hi('BUI THI NGAN')
    #2
    oddNumber();
    #3b
    Sum();
    #4a
    myDict();
    #5
    courses();
    #6
    Consonants();
    #7
    dividedNumber()
    #8
    lambdaSortList()
    #9
    UseFile();

    #--------------------------
    #1
    print("Tổng 2 số :", SumTwoNumber(5,7))
    #2
    matrix();
    #3
    newMatrix()
    #5
    Norm()
    #6
    Compute()
    #7
    dotMatrix()
    #8
    rank_shape_matrix()
    #9
    randomMatrix()
    #10
    maTranDonVi()
    #11
    matrix11()
    #12
    diagonalMatrix()
    #13
    DinhThucMaTran()
    #14
    matrix14()
    #15
    #VeBinhPhuong()
    #16
    bai16()
    #17
    #bai17()

    #18
    #bai18()

    #19
    #bai19()

    #20
    bai20()

