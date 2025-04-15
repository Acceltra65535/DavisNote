void mergeSort(int A[], int left, int right){
    if left < right{
        mid = (left + right) / 2;
        mergeSort(A, left, mid); //Recursively sort left half
        mergeSort(A, mid + 1, right); // Recursively sort right half
        mergeSort(A, mid, right); // Merge the two sorted halves
    }
}

void merge(int a[], int left, int mid, int right){
    create temporary arrys for left and right;
    copy data to temporary arrays;
    merge the temporary arrays back into the original array;
    copy any remaining elements of the temporary arrays;
}

// O(nlogn)