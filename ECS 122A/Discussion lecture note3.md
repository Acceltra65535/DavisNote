Jay. OH: M,W 4pm-5:30pm
         F 10am-noon.

Create a divide-and-conquer algorithm that finds the product of an array of integers with y=4 recursive calls. Given A=[4,3,5], the algorithm would return 60.

def product_DC(int A[]){
    n = A.size()
    //Base case, conquer
    if(n = 4){
        product = A[0], A[1]...A[n]
        return product
    }
    //Divide
    A = product_DC(A[0..n/4])
    B = product_DC(A[n/4+1...n/2])
    C = product_DC(A[n/2+1...3n/4])
    D = product_DC(A[3n/4+1...n])
    //4T(n/4)
    return A, B, C, D
        }

    //T(n) = 4T(n/4)+O(n)