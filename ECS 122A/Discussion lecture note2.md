void Foo(int A[]){

    let n = A.size();
    for (i i = 1 to n)
    {
        Boo(i);
    } // end for

    j = 1;
    
    while (j < n)
    {
        for (k = 0; k < n; k = k + 4)
        {
            print "hello";
        }
        j = j * y;
    }
    // end while
    
}

foo(int A[]){
    n = A.size;
    while (n > 1){

        n = n - 1;
        count++;
    }
    n = A.size;
    mergeSort(A);
    foo(A[1...n/y]);
    foo(A[n/y + 1...2n/y]);
}