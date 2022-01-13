#include <math.h>
#include "csparse.h"

typedef struct {
    cs* a;
    cs* h;
    double* b;
    double* c;
    double* x;
} SparseLP;

typedef struct {
    double* data;
    int m, n;
} DenseMatrix;

typedef struct {
    double* data;
    int m, n;
    int block_m, block_n;
} BlockedMatrix;

void dense_matrix_multiply(DenseMatrix* a, DenseMatrix* b, DenseMatrix* c) {
    if(a->n != b->m) return;
    int m = a->m;
    int n = b->m;
    int o = b->n;
    c->m = m;
    c->n = o;
    for(int j = 0; j < o; j++) {
        for(int i = 0; i < m; i++) {
            for(int k = 0; k < n; k++) {
                c->data[m*j + i] += a->data[m*j + k] * b->data[n*k + i];
            }
        }
    }
}

void test_data(SparseLP* lp) {
    cs* ta = cs_spalloc(2, 5, 10, 10, 1);
    // 3 2 1 1 0
    cs_entry(ta, 0, 0, 3.0);
    cs_entry(ta, 0, 1, 2.0);
    cs_entry(ta, 0, 2, 1.0);
    cs_entry(ta, 0, 3, 1.0);
    cs_entry(ta, 0, 4, 0.0);
    
    // 2 5 3 0 1
    cs_entry(ta, 1, 0, 2.0);
    cs_entry(ta, 1, 1, 5.0);
    cs_entry(ta, 1, 2, 3.0);
    cs_entry(ta, 1, 3, 0.0);
    cs_entry(ta, 1, 4, 1.0);
    lp->a = cs_triplet(ta);
    lp->h = cs_triplet(ta);
    cs_spfree(ta);

    lp->b = cs_calloc(2, sizeof(double));
    lp->b[0] = 10.0;
    lp->b[1] = 15.0;
    lp->c = cs_calloc(5, sizeof(double));
    lp->c[0] = -2.0;
    lp->c[1] =  3.0;
    lp->c[2] =  4.0;
    lp->c[3] =  0.0;
    lp->c[4] =  0.0;
    lp->x = cs_calloc(5, sizeof(double));
    lp->x[0] =  1.0;
    lp->x[1] =  1.0;
    lp->x[2] =  1.0;
    lp->x[3] =  4.0;
    lp->x[4] =  5.0;
}


void lp_deinit(SparseLP* lp) {
    cs_free(lp->b);
    cs_free(lp->c);
    cs_free(lp->x);
    cs_spfree(lp->a);
    cs_spfree(lp->h);
}

// 'b = a * diag(x)', 'b' is assumed to have same layout as 'a'
int cs_scale_columns(cs* A, cs* B, double* x) {
    if(!A || !B) return 0;
    int m = A->m;
    int n = A->n;
    if(m != B->m || n != B->n) return 0;

    int* Ap = A->p;
    double* Ax = A->x;
    double* Bx = B->x;
    
    for(int j = 0; j < n; j++) {
        for(int p = Ap[j]; p < Ap[j + 1]; p++) {
            Bx[p] = Ax[p] * x[j];
        }
    }
        
    return 1;
}

void cs_debug_print(cs* a) {
    int m = a->m;
    int n = a->n;
    double x[m*n];
    int* Ai = a->i;
    int* Ap = a->p;
    double* Ax = a->x;
    for(int i = 0; i < m*n; i++) x[i] = 0.0;
    for(int j = 0; j < n; j++) {
        for(int p = Ap[j]; p < Ap[j + 1]; p++) {
            x[Ai[p]*n + j] = Ax[p];
        }
    }
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            printf("%6.3f ", x[i*n + j]);
        }
        printf("\n");
    }
    printf("\n");
}


void lp_debug_print(SparseLP* lp) {
    cs* a = lp->a;
    int m = a->m;
    int n = a->n;
    double x[m*n];
    int* Ai = a->i;
    int* Ap = a->p;
    double* Ax = a->x;
    for(int i = 0; i < m*n; i++) x[i] = 0.0;
    for(int j = 0; j < n; j++) {
        for(int p = Ap[j]; p < Ap[j + 1]; p++) {
            x[Ai[p]*n + j] = Ax[p];
        }
    }
    for(int j = 0; j < n; j++) printf("%6.3f ", lp->c[j]);
    printf("|\n");
    for(int j = 0; j < n; j++) printf("-------");
    printf("+-------\n");
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            printf("%6.3f ", x[i*n + j]);
        }
        printf("| %6.3f\n", lp->b[i]);
    }
    printf("\n");
}



int main(void) {
    SparseLP linear_program;
    SparseLP* lp = &linear_program;
    test_data(lp);
    lp_debug_print(lp);
    double eps = 0.0000001;
    double r = 0.9;

    int m = lp->a->m;
    int n = lp->a->n;    
    double* s = cs_calloc(n, sizeof(double));
    double* w1 = cs_calloc(n, sizeof(double));
    double* w2 = cs_calloc(n, sizeof(double));
    double* w3 = cs_calloc(n, sizeof(double));
    double* w4 = cs_calloc(m, sizeof(double));

    css* schol = NULL;
    for(int i = 0; i < 1000; i++) {
        for(int j = 0; j < n; j++) {
            w3[j] = 0;
        }
        for(int j = 0; j < m; j++) {
            w4[j] = 0;
        }
        
        cs_gaxpy(lp->a, lp->x, w3);

        for(int j = 0; j < n; j++) {
            s[j] = lp->x[j] * lp->c[j];
        }
        cs_scale_columns(lp->a, lp->h, lp->x);
        cs_gaxpy(lp->h, s, w4);
        
        cs* ht = cs_transpose(lp->h, 1);
        cs* hht = cs_multiply(lp->h, ht);

        if(schol == NULL) schol = cs_schol(hht, 0);
        csn* hht_chol = cs_chol(hht, schol);
        cs_ipvec(m, schol->Pinv, w4, w1); /* w1 = P*w4   */
        cs_lsolve(hht_chol->L, w1);       /* w1 = L\w1  */
        cs_ltsolve(hht_chol->L, w1);      /* w1 = L'\w1 */
        cs_pvec(n, schol->Pinv, w1, w2);  /* w2 = P'*w1 */
        for(int j = 0; j < n; j++) {
            s[j] = -s[j];
        }
        cs_gaxpy(ht, w2, s);
        double err = 0.0;
        for(int j = 0; j < n; j++) {
            err = fmax(err, fabs(s[j]));
        }
        printf("%3i | %f | ", i, err);
        double alpha = r / err;
        for(int j = 0; j < n; j++) {
            lp->x[j] = lp->x[j] - alpha * lp->x[j] * s[j];
            printf("%10.7f, ", lp->x[j]);
        }
        printf("\n");
        cs_nfree(hht_chol);
        cs_spfree(ht);
        cs_spfree(hht);
        if(err < eps) break;
    }


    cs_free(s);
    cs_free(w1);
    cs_free(w2);
    cs_free(w3);
    cs_free(w4);
    cs_sfree(schol);
    lp_deinit(lp);
}