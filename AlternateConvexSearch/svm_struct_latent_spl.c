/************************************************************************/
/*                                                                      */
/*   svm_struct_latent_spl.c                                            */
/*                                                                      */
/*   Main Optimization Code for Latent SVM^struct using Self-Paced      */
/*   Learning. NOTE: This implementation modifies the CCCP code by      */
/*   Chun-Nam Yu, specifically the file svm_struct_latent_cccp.c,       */
/*   which is a part of the Latent SVM^struct package available on      */
/*   Chun-Nam Yu's webpage.                                             */
/*                                                                      */
/*   Authors: M. Pawan Kumar and Ben Packer                             */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/

#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "debug.h"
#include "svm_struct_latent_api.h"
#include "./svm_light/svm_learn.h"
#include "./svm_light/svm_common.h"
#include "mosek_api.h"


#define ALPHA_THRESHOLD 1E-14
#define IDLE_ITER 20
#define CLEANUP_CHECK 50
#define STOP_PREC 1E-2
#define UPDATE_BOUND 3
#define MAX_CURRICULUM_ITER 10

#define MAX_OUTER_ITER 400

#define MAX(x,y) ((x) < (y) ? (y) : (x))
#define MIN(x,y) ((x) > (y) ? (y) : (x))

#define DEBUG_LEVEL 0

#define ASIGM -1.5
#define LOG2_E 0.69314718055994529

//#define SECOND_PROP 0.05
//#define DELAY 3


int mosek_qp_optimize(double**, double*, double*, long, double, double*);

void my_read_input_parameters(int argc, char* argv[], char *trainfile,char *modelfile, char *examplesfile, char *timefile, char *latentfile,char *slackfile, char *uncertaintyfile, char *noveltyfile, char*lossfile, char*fycachefile, char *difficultyfile, LEARN_PARM *learn_parm, KERNEL_PARM *kernel_parm,STRUCT_LEARN_PARM *struct_parm, double *init_spl_weight, double*spl_factor);

void my_wait_any_key();

int resize_cleanup(int size_active, int **ptr_idle, double **ptr_alpha, double **ptr_delta, DOC ***ptr_dXc,
		double ***ptr_G, int *mv_iter);

void approximate_to_psd(double **G, int size_active, double eps);

void Jacobi_Cyclic_Method(double eigenvalues[], double *eigenvectors, double *A, int n);

double log2 (double x)
{
  return log (x) / LOG2_E;
}

int compare_dbl (const void * a, const void * b)
{
  double c = *(double*)a;
  double d = *(double*)b;
  if (c < d)
    return -1;
  if (c == d)
    return 0;
  if (c > d)
    return 1; 
}

double array_max (double *array, int numElts)
{
  double max = array[0];
  int i;
  for (i=1; i<numElts; ++i)
    max = MAX(array[i], max);
  return max;
}

double array_min (double *array, int numElts)
{
  double min = array[0];
  int i;
  for (i=1; i<numElts; ++i)
    min = MIN(array[i], min);
  return min;
}

double array_median (double *array, int numElts)
{
  double * array_copy = calloc (numElts, sizeof (double));
  memcpy (array_copy, array, numElts);
  qsort (array_copy, numElts, sizeof (double), compare_dbl);
  
  return array[(numElts-1) / 2];
}

double sprod_nn(double *a, double *b, long n) {
  double ans=0.0;
  long i;
  for (i=1;i<n+1;i++) {
    ans+=a[i]*b[i];
  }
  return(ans);
}

void add_vector_nn(double *w, double *dense_x, long n, double factor) {
  long i;
  for (i=1;i<n+1;i++) {
    w[i]+=factor*dense_x[i];
  }
}

// log_svector (file *, "fycache for asdf: ", asdf);
void log_fycache (FILE *f, SVECTOR **fycache, int m, int iter)
{
  WORD *w;
  int i,j;
  
  for (i=0; i<m; ++i)
    {
      fprintf (f, "iter %d example %d:", iter, i);
      w = fycache[i]->words;
      for (j=0; w[j].wnum > 0; ++j)
        fprintf (f, " %d:%f", w[j].wnum, w[j].weight);
      fprintf (f, "\n");
    }
}

double* add_list_nn(SVECTOR *a, long totwords) 
     /* computes the linear combination of the SVECTOR list weighted
	by the factor of each SVECTOR. assumes that the number of
	features is small compared to the number of elements in the
	list */
{
    SVECTOR *f;
    long i;
    double *sum;

    sum=create_nvector(totwords);

    for(i=0;i<=totwords;i++) 
      sum[i]=0;

    for(f=a;f;f=f->next)  
      add_vector_ns(sum,f,f->factor);

    return(sum);
}


void find_most_violated_constraint(EXAMPLE *ex, LABEL *ybar, LATENT_VAR *hbar, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
  switch (sparm->margin_type) {
    case 0: find_most_violated_constraint_marginrescaling (ex, ybar, hbar, sm, sparm); break;
    case 1: find_most_violated_constraint_oppositey (ex, ybar, hbar, sm, sparm); break;
    default: printf ("Unrecognized margin_type '%d'\n", sparm->margin_type);
    exit(1);
  }
}


double current_obj_val(EXAMPLE *ex, SVECTOR **fycache, long m, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, double C, int *valid_examples) {

  long i, j;
  SVECTOR *f, *fy, *fybar, *lhs;
  LABEL       ybar;
  LATENT_VAR hbar;
  double lossval, margin;
  double *new_constraint;
	double obj = 0.0;

  /* find cutting plane */
  lhs = NULL;
  margin = 0;
  for (i=0;i<m;i++) {
		if(!valid_examples[i])
			continue;
    find_most_violated_constraint(&(ex[i]), &ybar, &hbar, sm, sparm);
    /* get difference vector */
    fy = copy_svector(fycache[i]);
    fybar = psi(ex[i].x,ybar,hbar,sm,sparm);
    lossval = loss(ex[i].y,ybar,hbar,sparm);

    /* scale difference vector */
    for (f=fy;f;f=f->next) {
      //f->factor*=1.0/m;
      f->factor*=ex[i].x.example_cost/m;
    }
    for (f=fybar;f;f=f->next) {
      //f->factor*=-1.0/m;
      f->factor*=-ex[i].x.example_cost/m;
    }
    /* add ybar to constraint */
    append_svector_list(fy,lhs);
    append_svector_list(fybar,fy);
    lhs = fybar;
    //margin+=lossval/m;
		margin += lossval*ex[i].x.example_cost/m;
  }

  /* compact the linear representation */
  new_constraint = add_list_nn(lhs, sm->sizePsi);
  free_svector(lhs);

	obj = margin;
	for(i = 1; i < sm->sizePsi+1; i++)
		obj -= new_constraint[i]*sm->w[i];
	if(obj < 0.0)
		obj = 0.0;
	obj *= C;
	for(i = 1; i < sm->sizePsi+1; i++)
		obj += 0.5*sm->w[i]*sm->w[i];
  free(new_constraint);

	return obj;
}

int compar(const void *a, const void *b)
{
  sortStruct *c = (sortStruct *) a;
  sortStruct *d = (sortStruct *) b;
  if(c->val < d->val)
    return -1;
  if(c->val > d->val)
    return 1;
  return 0;
}


SVECTOR* find_cutting_plane(EXAMPLE *ex, SVECTOR **fycache, double *margin, long m, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm,
														int *valid_examples) {

  long i, j;
  SVECTOR *f, *fy, *fybar, *lhs;
  LABEL       ybar;
  LATENT_VAR hbar;
  double lossval;
  double *new_constraint;
	long valid_count = 0;

  long l,k;
  SVECTOR *fvec;
  WORD *words;  

  /* find cutting plane */
  lhs = NULL;
  *margin = 0;

	for (i=0;i<m;i++) {
		if (valid_examples[i]) {
			valid_count++;
		}
	}

  for (i=0;i<m;i++) {

		if (!valid_examples[i]) {
			continue;
		}

    find_most_violated_constraint(&(ex[i]), &ybar, &hbar, sm, sparm);
    // printf ("Most violated constraint %d is (y, h) = (%d, %d)\n", i, ybar.label, hbar.position);
    /* get difference vector */
    fy = copy_svector(fycache[i]);
    fybar = psi(ex[i].x,ybar,hbar,sm,sparm);
    lossval = loss(ex[i].y,ybar,hbar,sparm);
    free_label(ybar);
    free_latent_var(hbar);
		
    /* scale difference vector */
    for (f=fy;f;f=f->next) {
      //f->factor*=1.0/m;
      //f->factor*=ex[i].x.example_cost/m;
      f->factor*=ex[i].x.example_cost/valid_count;
    }
    for (f=fybar;f;f=f->next) {
      //f->factor*=-1.0/m;
      //f->factor*=-ex[i].x.example_cost/m;
      f->factor*=-ex[i].x.example_cost/valid_count;
    }
    /* add ybar to constraint */
    append_svector_list(fy,lhs);
    append_svector_list(fybar,fy);
    lhs = fybar;
    //*margin+=lossval/m;
    //*margin+=lossval*ex[i].x.example_cost/m;
    *margin+=lossval*ex[i].x.example_cost/valid_count;
  }

  /* compact the linear representation */
  new_constraint = add_list_nn(lhs, sm->sizePsi);
  free_svector(lhs);

  l=0;
  for (i=1;i<sm->sizePsi+1;i++) {
    if (fabs(new_constraint[i])>1E-10) l++; // non-zero
  }
  words = (WORD*)my_malloc(sizeof(WORD)*(l+1)); 
  assert(words!=NULL);
  k=0;
  for (i=1;i<sm->sizePsi+1;i++) {
    if (fabs(new_constraint[i])>1E-10) {
      words[k].wnum = i;
      words[k].weight = new_constraint[i]; 
      k++;
    }
  }
  words[k].wnum = 0;
  words[k].weight = 0.0;
  fvec = create_svector(words,"",1);

  free(words);
  free(new_constraint);

  return(fvec); 
}

/* project weights to ball of radius 1/sqrt{lambda} */
void project_weights(double *w, int sizePsi, double lambda)
{
	double norm = 0.0;
	double projection_factor = 1.0;
	int i;
	for(i=0;i<=sizePsi;i++)
		norm += w[i]*w[i];
	norm = sqrt(norm);
	if(norm > 1/sqrt(lambda))
	{
		projection_factor = 1.0/(sqrt(lambda)*norm);
		for(i=0;i<=sizePsi;i++)
			w[i] *= projection_factor;
	}
}

long *randperm(long m, long n)
{
	long *perm, *map;
	long i,j;

	if(m < n)
		n = m;
  perm = (long *) malloc(sizeof(long)*n);
	if(m == n) {
		for(i = 0; i < m; i++)
			perm[i] = i;
		return perm;
	}
  map = (long *) malloc(sizeof(long)*m);
  for(i = 0; i < m; i++)
    map[i] = i;
  for(i = 0; i < n; i++)
  {
    int r = (int) (((double) m-i)*((double) rand())/(RAND_MAX+1.0));
    perm[i] = map[r];
    for(j = r; j < m-1; j++)
      map[j] = map[j+1];
  }
  free(map);
  return perm;
}

/* stochastic subgradient descent for solving the convex structural SVM problem */
double stochastic_subgradient_descent(double *w, long m, int MAX_ITER, double C, double epsilon, SVECTOR **fycache, EXAMPLE *ex, 
															STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, int *valid_examples) {

	/* constants */
	int subset_size = 10;

	long *valid_indices;
	long num_valid = 0;
	long *perm;

	int iter, i;
	double learn_rate, lambda = 1.0/C;
	int is_valid, example_index;
  SVECTOR *fy, *fybar;
  LABEL       ybar;
  LATENT_VAR hbar;
  double lossval, primal_obj;
	double *new_w = (double *) my_malloc((sm->sizePsi+1)*sizeof(double));

  printf("Running stochastic structural SVM solver: "); fflush(stdout); 

	valid_indices = (long *) my_malloc(m*sizeof(long));
	for(i=0;i<m;i++) {
		if(valid_examples[i]) {
			valid_indices[num_valid] = i;
			num_valid++;
		}
	}
	if(num_valid < subset_size)
		subset_size = num_valid;

	/* initializations */
	iter = 0;
  srand(time(NULL));
	clear_nvector(w,sm->sizePsi);

	while(iter<MAX_ITER) {

		printf("."); fflush(stdout);

		/* learning rate for iteration */
		iter+=1;
		learn_rate = 1.0/(lambda*iter);

		for(i=0;i<=sm->sizePsi;i++)
			new_w[i] = (1.0-learn_rate*lambda)*w[i];

		/* randomly select a subset of examples */
		perm = randperm(num_valid,subset_size);

		for(i=0;i<subset_size;i++) {
			/* find subgradient */
   		find_most_violated_constraint(&(ex[valid_indices[perm[i]]]), &ybar, &hbar, sm, sparm);
   		lossval = loss(ex[valid_indices[perm[i]]].y,ybar,hbar,sparm);
   		fy = copy_svector(fycache[valid_indices[perm[i]]]);
   		fybar = psi(ex[valid_indices[perm[i]]].x,ybar,hbar,sm,sparm);
	
			/* update weight vector */
			/* ignoring example cost for simplicity */
			add_vector_ns(new_w,fy,ex[valid_indices[perm[i]]].x.example_cost*learn_rate/subset_size);
			add_vector_ns(new_w,fybar,-ex[valid_indices[perm[i]]].x.example_cost*learn_rate/subset_size);

			/* free variables */
   		free_label(ybar);
   		free_latent_var(hbar);
			free_svector(fy);
			free_svector(fybar);
		}

		free(perm);

		for(i=0;i<=sm->sizePsi;i++)
			w[i] = new_w[i];
		/* optional step: project weights to ball of radius 1/sqrt{lambda} */
		project_weights(w,sm->sizePsi,lambda);

	}

	free(valid_indices);
	free(new_w);

  printf(" Inner loop optimization finished.\n"); fflush(stdout); 

	/* return primal objective value */
	primal_obj = current_obj_val(ex, fycache, m, sm, sparm, C, valid_examples);
	return(primal_obj);

}

double cutting_plane_algorithm(double *w, long m, int MAX_ITER, double C, double epsilon, SVECTOR **fycache, EXAMPLE *ex, 
															STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, int *valid_examples) {
  long i,j;
  double *alpha;
  DOC **dXc; /* constraint matrix */
  double *delta; /* rhs of constraints */
  SVECTOR *new_constraint;
  int iter, size_active; 
  double value;
	double threshold = 0.0;
  double margin;
  double primal_obj, cur_obj;
	double *cur_slack = NULL;
	int mv_iter;
	int *idle = NULL;
	double **G = NULL;
	SVECTOR *f;
	int r;

  /* set parameters for hideo solver */
  LEARN_PARM lparm;
  KERNEL_PARM kparm;
  MODEL *svm_model=NULL;
  lparm.biased_hyperplane = 0;
  lparm.epsilon_crit = MIN(epsilon,0.001);
  lparm.svm_c = C;
  lparm.sharedslack = 1;
  kparm.kernel_type = LINEAR;

  lparm.remove_inconsistent=0;
  lparm.skip_final_opt_check=0;
  lparm.svm_maxqpsize=10;
  lparm.svm_newvarsinqp=0;
  lparm.svm_iter_to_shrink=-9999;
  lparm.maxiter=100000;
  lparm.kernel_cache_size=40;
  lparm.eps = epsilon; 
  lparm.transduction_posratio=-1.0;
  lparm.svm_costratio=1.0;
  lparm.svm_costratio_unlab=1.0;
  lparm.svm_unlabbound=1E-5;
  lparm.epsilon_a=1E-10;  /* changed from 1e-15 */
  lparm.compute_loo=0;
  lparm.rho=1.0;
  lparm.xa_depth=0;
  strcpy(lparm.alphafile,"");
  kparm.poly_degree=3;
  kparm.rbf_gamma=1.0;
  kparm.coef_lin=1;
  kparm.coef_const=1;
  strcpy(kparm.custom,"empty");
 
  iter = 0;
  size_active = 0;
  alpha = NULL;
  dXc = NULL;
  delta = NULL;

  printf("Running structural SVM solver: "); fflush(stdout); 

	new_constraint = find_cutting_plane(ex, fycache, &margin, m, sm, sparm, valid_examples);
  
  // printf ("Found the following first constraint:\n");
  // print_svec (new_constraint);
  
 	value = margin - sprod_ns(w, new_constraint);
	while((value>threshold+epsilon)&&(iter<MAX_ITER)) {
		iter+=1;
		size_active+=1;

		printf("."); fflush(stdout); 


    /* add  constraint */
  	dXc = (DOC**)realloc(dXc, sizeof(DOC*)*size_active);
   	assert(dXc!=NULL);
   	dXc[size_active-1] = (DOC*)malloc(sizeof(DOC));
   	dXc[size_active-1]->fvec = new_constraint; 
   	dXc[size_active-1]->slackid = 1; // only one common slackid (one-slack)
   	dXc[size_active-1]->costfactor = 1.0;

   	delta = (double*)realloc(delta, sizeof(double)*size_active);
   	assert(delta!=NULL);
   	delta[size_active-1] = margin;

   	alpha = (double*)realloc(alpha, sizeof(double)*size_active);
   	assert(alpha!=NULL);
   	alpha[size_active-1] = 0.0;

		idle = (int *) realloc(idle, sizeof(int)*size_active);
		assert(idle!=NULL);
		idle[size_active-1] = 0;

		/* update Gram matrix */
		G = (double **) realloc(G, sizeof(double *)*size_active);
		assert(G!=NULL);
		G[size_active-1] = NULL;
		for(j = 0; j < size_active; j++) {
			G[j] = (double *) realloc(G[j], sizeof(double)*size_active);
			assert(G[j]!=NULL);
		}
		for(j = 0; j < size_active-1; j++) {
			G[size_active-1][j] = sprod_ss(dXc[size_active-1]->fvec, dXc[j]->fvec);
			G[j][size_active-1]  = G[size_active-1][j];
		}
		G[size_active-1][size_active-1] = sprod_ss(dXc[size_active-1]->fvec,dXc[size_active-1]->fvec);

		/* hack: add a constant to the diagonal to make sure G is PSD */
		G[size_active-1][size_active-1] += 1e-6;

   	/* solve QP to update alpha */
		r = mosek_qp_optimize(G, delta, alpha, (long) size_active, C, &cur_obj);
    /*
    double eps = 1e-12;
    while(r >= 1293 && r <= 1296 && eps<100)
    {
        printf("|"); fflush(stdout);
        //approximate_to_psd(G,size_active,eps);
        for(j = 0; j < size_active; j++)
            if(eps > 1e-12)
                G[j][j] += eps - eps/100.0;
            else
                G[j][j] += eps;
        r = mosek_qp_optimize(G, delta, alpha, (long) size_active, C, &cur_obj);
        eps *= 100.0;
    }
    // undo changes to G
    if(eps > 1e-12)
        for(j = 0; j < size_active; j++)
    G[j][j] -= eps/100.0;
    */
		if(r >= 1293 && r <= 1296)
		{
			printf("r:%d. G might not be psd due to numerical errors.\n",r);
			exit(1);
		}
		else if(r)
		{
			printf("Error %d in mosek_qp_optimize: Check ${MOSEKHOME}/${VERSION}/tools/platform/${PLATFORM}/h/mosek.h\n",r);
			exit(1);
		}

   	clear_nvector(w,sm->sizePsi);
   	for (j=0;j<size_active;j++) {
     	if (alpha[j]>C*ALPHA_THRESHOLD) {
				add_vector_ns(w,dXc[j]->fvec,alpha[j]);
				idle[j] = 0;
     	}
			else
				idle[j]++;
   	}

		cur_slack = (double *) realloc(cur_slack,sizeof(double)*size_active);

		for(i = 0; i < size_active; i++) {
			cur_slack[i] = 0.0;
			for(f = dXc[i]->fvec; f; f = f->next) {
				j = 0;
				while(f->words[j].wnum) {
					cur_slack[i] += w[f->words[j].wnum]*f->words[j].weight;
					j++;
				}
			}
			if(cur_slack[i] >= delta[i])
				cur_slack[i] = 0.0;
			else
				cur_slack[i] = delta[i]-cur_slack[i];
		}

		mv_iter = 0;
		if(size_active > 1) {
			for(j = 0; j < size_active; j++) {
				if(cur_slack[j] >= cur_slack[mv_iter])
					mv_iter = j;
			}
		}

		if(size_active > 1)
			threshold = cur_slack[mv_iter];
		else
			threshold = 0.0;

 		new_constraint = find_cutting_plane(ex, fycache, &margin, m, sm, sparm, valid_examples);
    // printf ("Found the following constraint %d:\n", iter);
    //     print_svec (new_constraint);
 		
   	value = margin - sprod_ns(w, new_constraint);

		if((iter % CLEANUP_CHECK) == 0)
		{
			printf("+"); fflush(stdout);
			size_active = resize_cleanup(size_active, &idle, &alpha, &delta, &dXc, &G, &mv_iter);
		}

 	} // end cutting plane while loop 

	primal_obj = current_obj_val(ex, fycache, m, sm, sparm, C, valid_examples);

  printf(" Inner loop optimization finished.\n"); fflush(stdout); 
      
  /* free memory */
  for (j=0;j<size_active;j++) {
		free(G[j]);
    free_example(dXc[j],0);	
  }
	free(G);
  free(dXc);
  free(alpha);
  free(delta);
  free_svector(new_constraint);
	free(cur_slack);
	free(idle);
  if (svm_model!=NULL) free_model(svm_model,0);

  return(primal_obj);
}

int check_acs_convergence(int *prev_valid_examples, int *valid_examples, long m)
{
	long i;
	int converged = 1;

	for (i=0;i<m;i++) {
		if (prev_valid_examples[i] != valid_examples[i]) {
			converged = 0;
			break;
		}
	}

	return converged;
}

/*int update_valid_examples(double *w, long m, double C, SVECTOR **fycache, EXAMPLE *ex, 
													STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, int *valid_examples, double spl_weight) {

	long i, j;

	/* if self-paced learning weight is non-positive, all examples are valid 
	if(spl_weight <= 0.0) {
		for (i=0;i<m;i++)
			valid_examples[i] = 1;
		return (m);
	}

	sortStruct *slack = (sortStruct *) malloc(m*sizeof(sortStruct));
	LABEL ybar;
	LATENT_VAR hbar;
	SVECTOR *f, *fy, *fybar;
	double lossval;
	double penalty = 1.0/spl_weight;
	if(penalty < 0.0)
		penalty = DBL_MAX;

	for (i=0;i<m;i++) {
		find_most_violated_constraint(ex[i].x, ex[i].y, &ybar, &hbar, sm, sparm);
		fy = copy_svector(fycache[i]);
		fybar = psi(ex[i].x,ybar,hbar,sm,sparm);
		slack[i].index = i;
		slack[i].val = loss(ex[i].y,ybar,hbar,sparm);
		for (f=fy;f;f=f->next) {
			j = 0;
			while (1) {
				if(!f->words[j].wnum)
					break;
				slack[i].val -= sm->w[f->words[j].wnum]*f->words[j].weight;
				j++;
			}
		}
		for (f=fybar;f;f=f->next) {
			j = 0;
			while (1) {
				if(!f->words[j].wnum)
					break;
				slack[i].val += sm->w[f->words[j].wnum]*f->words[j].weight;
				j++;
			}
		}
		free_svector(fy);
		free_svector(fybar);
	}
	qsort(slack,m,sizeof(sortStruct),&compar);

	int nValid = 0;
	for (i=0;i<m;i++)
		valid_examples[i] = 0;
	for (i=0;i<m;i++) {
		if(slack[i].val*C/m > penalty)
			break;
		valid_examples[slack[i].index] = 1;
		nValid++;
	}

	free(slack);

	return nValid;
}*/

/* Convert a single one-indexed sparse vector into a full vector of doubles. */
double *convert_from_svector(SVECTOR *svec,int size) {
  double *v = calloc (1, sizeof(double)*size);
  SVECTOR *f;
  long j;
	int pos;
	
	f = svec;
	j = 0;
	pos = f->words[j].wnum;
	while (pos)
		{
			assert (pos <= size);
			v[pos-1] = f->words[j].weight;
			++j;
			pos = f->words[j].wnum;			
		}

  return v;
}

double get_novelty(EXAMPLE *ex, long exNum, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
  printf("*"); fflush(stdout);
  int numPairs;  
  long j;

	SVECTOR *psi_h_star_sparse;
	SVECTOR **psi_h_y_hats_sparse;
	
	double *psi_h_star;
	double **psi_h_y_hats;
  
  psi_h_star_sparse = psi(ex[exNum].x, ex[exNum].y, ex[exNum].h, sm, sparm);
  psi_h_star = convert_from_svector(psi_h_star_sparse,sm->sizePsi);

  psi_h_y_hats_sparse = get_all_psi(ex, exNum, &numPairs, sm, sparm);
  psi_h_y_hats = malloc(numPairs*sizeof(double *));

  for(j=0;j<numPairs;j++) {
    psi_h_y_hats[j] = convert_from_svector(psi_h_y_hats_sparse[j],sm->sizePsi);
  }

	double losses[numPairs];
  get_all_losses(ex, exNum, losses, sm, sparm);
  
  // printf ("Getting novelty for example %d\n", (int)exNum);
  double novelty = compute_delta_w(sm->w,psi_h_star,psi_h_y_hats,losses,sm->sizePsi,numPairs);
  // double novelty = 0.0;
  // printf ("Computed novelty %f for example %d\n", novelty, (int)exNum);
  
  for(j=0;j<numPairs;j++) {
    free_svector(psi_h_y_hats_sparse[j]);
		free(psi_h_y_hats[j]);
  }  
  free(psi_h_y_hats_sparse);
  free(psi_h_y_hats);

  free_svector(psi_h_star_sparse);  
  free(psi_h_star);

	return novelty;
}

double get_entropy(double *distrib, int numEntries) {
  int k;
  double p, entropy;
  
  entropy = 0.0;
  for(k=0;k<numEntries;k++) {
      p = distrib[k];
      if(p) {
	      entropy += p*log(p);
      }
  } 
  entropy *= -1;   
	return(entropy);
}

// Get the weight of a generalized probability distribution (weight <= 1)
double
get_weight (double *probs, int numEntries)
{
  int k;
  double weight;
  
  weight = 0.0;
  for (k=0; k<numEntries; ++k)
    weight += probs[k];

	return weight;
}

double
get_renyi_entropy (double *probs, double alpha, int numEntries)
{
  int k;
  double p, entropy;
  
  if (alpha == 1)
    {
      entropy = 0.0;
      for (k=0; k<numEntries; ++k)
        {          
          p = probs[k];
          if (p > 0)
    	      entropy -= p * log2 (p);
        }
      entropy /= get_weight (probs, numEntries);
    }
  else if (alpha > 1)
    {
      double pMax, sum, term1, term2, term3;
      
      pMax = array_max (probs, numEntries);
      
      sum = 0.0;
      for (k=0; k<numEntries; ++k)
        sum += pow ((probs[k] / pMax), alpha);
        
      term1 = alpha * log2 (pMax);      
      term2 = log2 (sum);
      term3 = log2 (get_weight (probs, numEntries));
      
      entropy = (term1 + term2 - term3) / (1 - alpha);
    }
  else
    {
      printf ("WARNING: called get_renyi_entropy for unsupported alpha = %f\n", alpha);
    }
  
  return entropy;
}

double * get_h_probabilities(PATTERN x, LABEL y, int numPositions, double Asigm, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
  double * hvScores = malloc(numPositions * sizeof(double));
  get_latent_variable_scores(x, y, hvScores, sm, sparm); 

  //just to be safe
  if (y.label == -1) {
    hvScores[0] = 1.0;
    return hvScores;
  }
  double scoreSum = 0.0;
  long j;
  
  //// TODO: remove
  //double max_score = -1E10;
  //int max_pos = -1;
  
  // double minScore = array_min (hvScores, numPositions);
  // double medianScore = array_median (hvScores, numPositions);
  // double maxScore = array_max (hvScores, numPositions);
  // double tmp;
    
  for(j = 0; j < numPositions; j++) {
    
     // SIGMOID or BOLTZMANN if const is 1 or 0
     hvScores[j] = 1/(1+exp(Asigm*hvScores[j]));
     scoreSum += hvScores[j];
    
    // // SQUARED approach, scale invariant (next, subtract median)    
    // hvScores[j] = (hvScores[j] - minScore + 1E-10) * (hvScores[j] - minScore + 1E-10);
    // scoreSum += hvScores[j];
    
    // // RESCALING approach, scale invariant (next, subtract median) 
    // hvScores[j] = exp ((hvScores[j] - medianScore) / (maxScore - medianScore + 1E-10) * 10);
    // if (hvScores[j] == hvScores[j])
    //   scoreSum += hvScores[j];
    // else
    //   printf ("Whos your daddy\n"); // TODO: watch potty mouth
    
    //// TODO: remove
    //if (hvScores[j] > max_score) {
    //  max_score = hvScores[j];
    //  max_pos = j;
    //}
    
      ////TOP TWO
    //if (hvScores[j] > second_max_score) {
    //  if (hvScores[j] > max_score) {
      //second_max_score = max_score;
      //second_max_pos = max_pos;
      //max_score = hvScores[j];
      //max_pos = j;
      //} else {
      //	second_max_score = hvScores[j];
      //	second_max_pos = j;
      //}
      //}
  }

  for(j = 0; j < numPositions; j++) {
    hvScores[j] /= scoreSum;
  }
  
  // // TODO: remove
  //for(j = 0; j < numPositions; j++) {
  //  hvScores[j] = (j == max_pos ? 1 : 1E-10);
  //}

  //// TOP TWO
  //for (j = 0; j < numPositions; j++) {
  //  hvScores[j] = 0.0;
  //  if (j == max_pos) {
  //    hvScores[j] = 1.0 - SECOND_PROP;
  //  } else if (j == second_max_pos) {
  //    hvScores[j] = SECOND_PROP;
  //  }
  //}
  return hvScores;
}

SVECTOR * get_expected_psih(PATTERN x, LABEL y, int numPositions, double Asigm, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
  printf ("WARNING: get_expected_psih is DEPRECATED and may not work\n");
  double * hvScores;
	long j;
	LATENT_VAR h;
	
  if (y.label == -1)
    {
      h.position = -1;
      return psi(x, y, h, sm, sparm);
    }
	
	hvScores = get_h_probabilities(x, y, numPositions, Asigm, sm, sparm);
  // printf ("Just computed hv score %f for h=0\n", hvScores[0]);
	h.position = 0;
	SVECTOR * psih = psi(x, y, h, sm, sparm);
	SVECTOR * expected_psih = smult_s(psih, hvScores[0]);
	free_svector(psih);
	for (h.position = 1; h.position < numPositions; h.position++) {
		SVECTOR * b = psi(x, y, h, sm, sparm);
		SVECTOR * mult =  smult_s(b, hvScores[h.position]);
    SVECTOR * ma = add_ss(expected_psih, mult);
    
    free_svector(mult);
		free_svector(expected_psih);
		free_svector(b);
		expected_psih = ma;  
	}
	free(hvScores);
	return expected_psih;
}

sortStruct *get_example_scores(long m, double C, SVECTOR **fycache, EXAMPLE *ex, 
													STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, 
                          double *losses, double *slacks, double *entropies, double *novelties, double *difficulties) {
	long i, j;
  int numPositions;
	double difficulty, lossval, uncertainty, novelty, *hvScores, scoreSum;

	double uncertaintyWeight = sparm->uncertainty_weight;
  double noveltyWeight = sparm->novelty_weight;
  double difficultyWeight = 1.0-uncertaintyWeight-noveltyWeight;

  sortStruct *exampleScores = (sortStruct *) malloc(m*sizeof(sortStruct));
	LABEL ybar;
	LATENT_VAR hbar;
	SVECTOR *f, *fy, *fybar;

	for (i=0;i<m;i++) {		
		find_most_violated_constraint(&(ex[i]), &ybar, &hbar, sm, sparm);
		fy = copy_svector(fycache[i]);
		fybar = psi(ex[i].x,ybar,hbar,sm,sparm);
		exampleScores[i].index = i;
		lossval = loss(ex[i].y,ybar,hbar,sparm);
		
		difficulty = 0.0;
		uncertainty = 0.0;
		novelty = 0.0;
		
		
		if (sparm->renyi_exponent == -1.0)
		  {
		    for (f=fy;f;f=f->next) {
    			j = 0;
    			while (1) {
    				if(!f->words[j].wnum)
    					break;
    				difficulty -= sm->w[f->words[j].wnum]*f->words[j].weight;
    				j++;
    			}
    		}
    		for (f=fybar;f;f=f->next) {
    			j = 0;
    			while (1) {
    				if(!f->words[j].wnum)
    					break;
    				difficulty += sm->w[f->words[j].wnum]*f->words[j].weight;
    				j++;
    			}
    		}
    		
    		exampleScores[i].val = lossval + difficulty;
		  }
		else if (sparm->renyi_exponent >= 1.0)
      {
        // calculate the joint probs over yhat, hhat for point i
        // save correct and incorrectly labeled parts of distribution
        numPositions = get_num_latent_variable_options(ex[i].x, sm, sparm);
        double *correct_probs = calloc (numPositions, sizeof (double));
        double *incorrect_probs = calloc (numPositions, sizeof (double));
        get_yhat_hhat_probs (ex[i].x, ex[i].y, correct_probs, incorrect_probs, sm, sparm);

        // compute entropy of each half
        double correct_entropy = get_renyi_entropy (correct_probs, sparm->renyi_exponent, numPositions);
        double incorrect_entropy = get_renyi_entropy (incorrect_probs, sparm->renyi_exponent, numPositions);
        
        free (correct_probs);
        free (incorrect_probs);

        exampleScores[i].val = correct_entropy - incorrect_entropy;
      }
    else if (sparm->renyi_exponent == -2.0)
      {
        numPositions = get_num_latent_variable_options_HACK(ex[i].x, ex[i].y, sm, sparm);
  	    hvScores = get_h_probabilities(ex[i].x, ex[i].y, numPositions, ASIGM, sm, sparm);
        uncertainty = get_entropy(hvScores, numPositions);
        free(hvScores);

    		exampleScores[i].val = lossval + uncertainty;
      }
    else if (sparm->renyi_exponent > 1.0)
      printf ("ERROR: Renyi exponent %f not implemented yet\n", sparm->renyi_exponent);
    else
      printf ("ERROR: Invalid renyi exponent %f\n", sparm->renyi_exponent);

    // if(noveltyWeight || sparm->print_extensive) {
    //   novelty = get_novelty(ex,i,sm,sparm);
    // } else {
    //   novelty = 0.0;    
    // }


      
    // exampleScores[i].val = (uncertaintyWeight + difficultyWeight)*lossval + difficultyWeight * difficulty + uncertaintyWeight * uncertainty + noveltyWeight * novelty; //score!!
    
    if(losses) losses[i] = lossval;
    if(slacks) slacks[i] = difficulty;
    if(entropies) entropies[i] = uncertainty;
    if(novelties) novelties[i] = novelty; 
    if(difficulties) difficulties[i] = exampleScores[i].val; 

		free_svector(fy);
		free_svector(fybar);
	}

  // print_mosek_stats ();

	qsort(exampleScores,m,sizeof(sortStruct),&compar);
  return(exampleScores);
}

int update_valid_examples(double *w, long m, double C, SVECTOR **fycache, EXAMPLE *ex, 
													STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, int *valid_examples, double spl_weight, 
                          double *losses, double *slacks, double *entropies, double *novelties, double *difficulties) {

  long i;
	/* if self-paced learning weight is non-positive, all examples are valid */
	if(spl_weight <= 0.0) {
		for (i=0;i<m;i++)
			valid_examples[i] = 1;
		return (m);
	}

	sortStruct *exampleScores = get_example_scores(m, C, fycache, ex, sm, sparm, losses, slacks, entropies, novelties, difficulties);

  double penalty = 1.0/spl_weight;
	if(penalty < 0.0)
		penalty = DBL_MAX;

	int nValid = 0;
	for (i=0;i<m;i++)
		valid_examples[i] = 0;
	for (i=0;i<m;i++) {
		if(exampleScores[i].val*C/m > penalty)
			break;
		valid_examples[exampleScores[i].index] = 1;
    //printf("%d ",ex[exampleScores[i].index].y.label);

		nValid++;
	}
  //printf("\n");
	free(exampleScores);

	return nValid;
}

double get_init_spl_weight(long m, double C, SVECTOR **fycache, EXAMPLE *ex, 
													 STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {

	sortStruct *exampleScores = get_example_scores(m, C, fycache, ex, sm, sparm, NULL, NULL, NULL, NULL, NULL);
  printf("\n");  

  long i;
  int half, totalPos, numPos;
  double uncertaintyWeight = sparm->uncertainty_weight;
  double noveltyWeight = sparm->novelty_weight;

	half = (int) round(sparm->init_valid_fraction*m);
	double init_spl_weight = (double)m/C/exampleScores[half].val;

  // if(sparm->init_valid_fraction_pos) { 
  //   totalPos = (int) round(sparm->init_valid_fraction_pos*m/2); //fraction of positive examples
  //   numPos = 0;
  //   for(i=0;i<m;i++) {
  //     if(ex[exampleScores[i].index].y.label == 1) numPos++;
  //     if(numPos >= totalPos) { 
  //       init_spl_weight = (double)m/C/exampleScores[i].val;
  //       break;
  //     }
  //   }
  // }

	free(exampleScores);

	return(init_spl_weight);
}

double alternate_convex_search(double *w, long m, int MAX_ITER, double C, double epsilon, SVECTOR **fycache, EXAMPLE *ex, 
                               STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, int *valid_examples, double spl_weight, 
                               double *losses, double *slacks, double *entropies, double *novelties, double *difficulties) {

	long i;
	int iter = 0, converged, nValid;
	double last_relaxed_primal_obj = DBL_MAX, relaxed_primal_obj, decrement;

	int *prev_valid_examples = (int *) malloc(m*sizeof(int));
	double *best_w = (double *) malloc((sm->sizePsi+1)*sizeof(double));

	for (i=0;i<sm->sizePsi+1;i++)
		best_w[i] = w[i];
	nValid = update_valid_examples(w, m, C, fycache, ex, sm, sparm, valid_examples, spl_weight, losses, slacks, entropies, novelties, difficulties);
	last_relaxed_primal_obj = current_obj_val(ex, fycache, m, sm, sparm, C, valid_examples);
	if(nValid < m)
		last_relaxed_primal_obj += (double)(m-nValid)/((double)spl_weight);

	for (i=0;i<m;i++) {
		prev_valid_examples[i] = 0;
	}

	for (iter=0;;iter++) {
		nValid = update_valid_examples(w, m, C, fycache, ex, sm, sparm, valid_examples, spl_weight, losses, slacks, entropies, novelties, difficulties);
		printf("ACS Iteration %d: number of examples = %d\n",iter,nValid); fflush(stdout);
		converged = check_acs_convergence(prev_valid_examples,valid_examples,m);
		if(converged) {
			break;
		}
		for (i=0;i<sm->sizePsi+1;i++)
			w[i] = 0.0;
		if(!sparm->optimizer_type)
			relaxed_primal_obj = cutting_plane_algorithm(w, m, MAX_ITER, C, epsilon, fycache, ex, sm, sparm, valid_examples);
		else
			relaxed_primal_obj = stochastic_subgradient_descent(w, m, MAX_ITER, C, epsilon, fycache, ex, sm, sparm, valid_examples);
		if(nValid < m)
			relaxed_primal_obj += (double)(m-nValid)/((double)spl_weight);
		decrement = last_relaxed_primal_obj-relaxed_primal_obj;
    printf("relaxed primal objective: %.4f\n", relaxed_primal_obj);
   	printf("decrement: %.4f\n", decrement); fflush(stdout);
		/*
		if (iter) {
    	printf("decrement: %.4f\n", decrement); fflush(stdout);
		}
		else {
			printf("decrement: N/A\n"); fflush(stdout);
		}
		*/
		if (decrement>=0.0) {
			for (i=0;i<sm->sizePsi+1;i++) {
				best_w[i] = w[i];
			}
		}
		if (decrement <= C*epsilon) {
			break;
		}
		last_relaxed_primal_obj = relaxed_primal_obj;
		for (i=0;i<m;i++) {
			prev_valid_examples[i] = valid_examples[i];
		}
	}

	for (i=0;i<m;i++) {
		prev_valid_examples[i] = 1;
	}

	if (iter) {
		for (i=0;i<sm->sizePsi+1;i++) {
			w[i] = best_w[i];
		}
	}

	double primal_obj;
	primal_obj = current_obj_val(ex, fycache, m, sm, sparm, C, prev_valid_examples);
	
	free(prev_valid_examples);
	free(best_w);

	return(primal_obj);
}


SAMPLE  generate_train_set(SAMPLE alldata, long *perm, int ntrain)
{
  SAMPLE  train;
  train.n = ntrain;
  long i;

  train.examples = (EXAMPLE *) malloc(train.n*sizeof(EXAMPLE));

  for(i = 0; i < train.n; i++)
  {
    train.examples[i] = alldata.examples[perm[i]];
  }

  return train;
}

SAMPLE  generate_validation_set(SAMPLE alldata, long *perm, int ntrain)
{
  SAMPLE  val;
  val.n = alldata.n - ntrain;
  long i;

  val.examples = (EXAMPLE *) malloc(val.n*sizeof(EXAMPLE));

  for(i = 0; i < val.n; i++)
  {
    val.examples[i] = alldata.examples[perm[ntrain+i]];
  }

  return val;
}

double compute_current_loss(SAMPLE val, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm)
{
	long i;
	LABEL y;
	LATENT_VAR h;
	double cur_loss = 0.0;
	double store;
	for(i = 0; i < val.n; i++)
	{
		classify_struct_example(val.examples[i].x,&y,&h,sm,sparm);
		store = loss(val.examples[i].y,y,h,sparm);
		cur_loss += store;
	}

	cur_loss /= (double) val.n;
	return cur_loss;
}

int main(int argc, char* argv[]) {
  double *w; /* weight vector */
  int outer_iter;
  long m, i;
  double C, epsilon;
  LEARN_PARM learn_parm;
  KERNEL_PARM kernel_parm;
  char trainfile[1024];
  char modelfile[1024];
	char examplesfile[1024];
	char timefile[1024];
	char latentfile[1024];
  char slackfile[1024];
  char uncertaintyfile[1024];
  char noveltyfile[1024];
  char difficultyfile[1024];
  char lossfile[1024];
  char fycachefile[1024];
  int MAX_ITER;
  /* new struct variables */
  SVECTOR **fycache, *diff, *fy;
  EXAMPLE *ex;
	SAMPLE alldata;
  SAMPLE sample;
	SAMPLE val;
  STRUCT_LEARN_PARM sparm;
  STRUCTMODEL sm;
  
  double decrement;
  double primal_obj, last_primal_obj;
  double stop_crit; 
	char itermodelfile[2000];

	/* self-paced learning variables */
	double init_spl_weight;
	double spl_weight;
	double spl_factor;
	int *valid_examples;
     
  double *slacks, *entropies, *novelties, *losses, *difficulties;
  /* read input parameters */
	my_read_input_parameters(argc, argv, trainfile, modelfile, examplesfile, timefile, latentfile, slackfile, uncertaintyfile, noveltyfile, lossfile, fycachefile, difficultyfile,
                      &learn_parm, &kernel_parm, &sparm, &init_spl_weight, &spl_factor); 

  epsilon = learn_parm.eps;
  C = learn_parm.svm_c;
  MAX_ITER = learn_parm.maxiter;

  /* read in examples */
  alldata = read_struct_examples(trainfile,&sparm);
  int ntrain = (int) round(1.0*alldata.n); /* no validation set */
	if(ntrain < alldata.n)
	{
	 srand(time(NULL));
 	 long *perm = randperm(alldata.n,alldata.n);
 	 sample = generate_train_set(alldata, perm, ntrain);
 	 val = generate_validation_set(alldata, perm, ntrain);
 	 free(perm);
	}
	else
	{
		sample = alldata;
	}
  ex = sample.examples;
  m = sample.n;
  
  /* initialization */
  init_struct_model(alldata,&sm,&sparm,&learn_parm,&kernel_parm); 

  w = create_nvector(sm.sizePsi);
  clear_nvector(w, sm.sizePsi);
  sm.w = w; /* establish link to w, as long as w does not change pointer */

  /* some training information */
  printf("C: %.8g\n", C);
	printf("spl weight: %.8g\n",init_spl_weight);
  printf("epsilon: %.8g\n", epsilon);
  printf("sample.n: %d\n", sample.n); 
  printf("sm.sizePsi: %ld\n", sm.sizePsi); fflush(stdout);
  
  slacks = calloc(m,sizeof(double));
  entropies = calloc(m,sizeof(double));
  novelties = calloc(m,sizeof(double));
  difficulties = calloc(m,sizeof(double));
  losses = calloc(m,sizeof(double));

  /* impute latent variable for first iteration */
  init_latent_variables(&sample,&learn_parm,&sm,&sparm);

  // {
  // int i;
  // for (i=0; i<sm.sizePsi; ++i)
  //   {
  //     printf ("h[%d] = %d", i, sample.examples[i].h.position);
  //   }
  // }

  // FILE *ffycache = fopen(fycachefile,"w");

  /* prepare feature vector cache for correct labels with imputed latent variables */
  fycache = (SVECTOR**)malloc(m*sizeof(SVECTOR*));
  for (i=0;i<m;i++) {
  	fy = psi(ex[i].x, ex[i].y, ex[i].h, &sm, &sparm);
  	diff = add_list_ss(fy);
  	free_svector(fy);
  	fy = diff;
  	fycache[i] = fy;
  }
  // log_fycache (ffycache, fycache, m, -3);

 	/* learn initial weight vector using all training examples */
	valid_examples = (int *) malloc(m*sizeof(int));
	if (init_spl_weight>0.0) {
		printf("INITIALIZATION\n"); fflush(stdout);
		for (i=0;i<m;i++) {
			valid_examples[i] = 1;
		}
		int initIter;
		for (initIter=0;initIter<2;initIter++) {
			if(!sparm.optimizer_type)
				primal_obj = cutting_plane_algorithm(w, m, MAX_ITER, C, epsilon, fycache, ex, &sm, &sparm, valid_examples);
			else
				primal_obj = stochastic_subgradient_descent(w, m, MAX_ITER, C, epsilon, fycache, ex, &sm, &sparm, valid_examples);
  		for (i=0;i<m;i++) {
   	 		free_latent_var(ex[i].h);
   	 		ex[i].h = infer_latent_variables(ex[i].x, ex[i].y, &sm, &sparm);
   		}
	    for (i=0;i<m;i++) {
  	    free_svector(fycache[i]);
        // if (sparm.using_argmax) {
    	    fy = psi(ex[i].x, ex[i].y, ex[i].h, &sm, &sparm);
     	    diff = add_list_ss(fy);
          free_svector(fy);
          fy = diff;
          fycache[i] = fy;
        // } else {
        //   fycache[i] = get_expected_psih(ex[i].x, ex[i].y, get_num_latent_variable_options_HACK(ex[i].x, ex[i].y, &sm, &sparm), ASIGM, &sm, &sparm);
        // }
    	}
      // log_fycache (ffycache, fycache, m, initIter-2);
		}
	}
	
  // {
  //   int i;
  //   for (i=0; i<sm.sizePsi; ++i)
  //     {
  //       // printf ("w[%d] = %f\n", i, w[i]);
  //       // printf ("sm.w[%d] = %f\n", i, sm.w[i]);
  //     }
  //   }
     
  /* outer loop: latent variable imputation */
  outer_iter = 0;
  last_primal_obj = DBL_MAX;
  decrement = 0;

  /* errors for validation set */

  double cur_loss, best_loss = DBL_MAX;
  int loss_iter;


	/* initializations */
	int latent_update = 0;
	FILE	*fexamples = fopen(examplesfile,"w");
	FILE	*ftime = fopen(timefile,"w");
	FILE	*flatent = fopen(latentfile,"w");
  FILE *fslack = fopen(slackfile,"w");
  FILE *fentropy = fopen(uncertaintyfile,"w");
  FILE *fnovelty = fopen(noveltyfile,"w");
  FILE *fdifficulty = fopen(difficultyfile,"w");
  FILE *floss = fopen(lossfile,"w");
	clock_t start = clock();

	spl_weight = init_spl_weight;
  while ((outer_iter<2)||((!stop_crit)&&(outer_iter<MAX_OUTER_ITER))) { 
		if(!outer_iter && init_spl_weight) {
			spl_weight = get_init_spl_weight(m, C, fycache, ex, &sm, &sparm);
      printf("Setting initial spl weight to %f\n",spl_weight);
		}
    printf("OUTER ITER %d\n", outer_iter); 
    /* cutting plane algorithm */
    //primal_obj = cutting_plane_algorithm(w, m, MAX_ITER, C, epsilon, fycache, ex, &sm, &sparm, valid_examples);
		/* solve biconvex self-paced learning problem */
		primal_obj = alternate_convex_search(w, m, MAX_ITER, C, epsilon, fycache, ex, &sm, &sparm, valid_examples, spl_weight, losses, slacks, entropies, novelties, difficulties);
		int nValid = 0;
		for (i=0;i<m;i++) {
			fprintf(fexamples,"%d ",valid_examples[i]);
			print_latent_var(ex[i].h,flatent);
      fprintf(fslack,"%f ",slacks[i]);
      fprintf(fentropy,"%f ",entropies[i]);
      fprintf(fnovelty,"%f ",novelties[i]);
      fprintf(fdifficulty,"%f ",difficulties[i]);
      fprintf(floss,"%f ",losses[i]);
			if(valid_examples[i]) {
				nValid++;
			}
		}
		fprintf(fexamples,"\n"); fflush(fexamples);
		fprintf(flatent,"\n"); fflush(flatent);
    fprintf(fslack,"\n"); fflush(fslack);
    fprintf(fentropy,"\n"); fflush(fentropy);
    fprintf(fnovelty,"\n"); fflush(fnovelty);
    fprintf(fdifficulty,"\n"); fflush(fdifficulty);
    fprintf(floss,"\n"); fflush(floss);
		clock_t finish = clock();
		fprintf(ftime,"%f %f\n",primal_obj,(((double)(finish-start))/CLOCKS_PER_SEC)); fflush(ftime);
    
    /* compute decrement in objective in this outer iteration */
    decrement = last_primal_obj - primal_obj;
    last_primal_obj = primal_obj;
    printf("primal objective: %.4f\n", primal_obj);
		if (outer_iter) {
    	printf("decrement: %.4f\n", decrement); fflush(stdout);
		}
		else {
			printf("decrement: N/A\n"); fflush(stdout);
		}
    
    stop_crit = (abs (decrement) < C*epsilon);
		/* additional stopping criteria */
		if(nValid < m)
			stop_crit = 0;
		if(!latent_update)
			stop_crit = 0;
 

		if(nValid) {
      for (i=0;i<m;i++) {
        /* impute latent variable using updated weight vector */
    /* (imputation happens even if imputed latent variables won't be used in inner loop, so that we can still use latent_update as a stopping criterion)*/
          free_latent_var(ex[i].h);
          ex[i].h = infer_latent_variables(ex[i].x, ex[i].y, &sm, &sparm);
			}
			latent_update++;
		}
  
    /* re-compute feature vector cache */
    for (i=0;i<m;i++) {
      free_svector(fycache[i]);
      if (sparm.using_argmax || outer_iter) {
        fy = psi(ex[i].x, ex[i].y, ex[i].h, &sm, &sparm);
        diff = add_list_ss(fy);
        free_svector(fy);
        fy = diff;
        fycache[i] = fy;
      } else {
        fycache[i] = get_expected_psih(ex[i].x, ex[i].y, get_num_latent_variable_options_HACK(ex[i].x, ex[i].y, &sm, &sparm), ASIGM, &sm, &sparm);
      }
    }
    // log_fycache (ffycache, fycache, m, outer_iter);
		sprintf(itermodelfile,"%s.%04d",modelfile,outer_iter);
		write_struct_model(itermodelfile, &sm, &sparm);

		if(ntrain < alldata.n) {
			cur_loss = compute_current_loss(val,&sm,&sparm);
			if(cur_loss <= best_loss) {
				best_loss = cur_loss;
				loss_iter = outer_iter;
			}
			printf("CURRENT LOSS: %f\n",cur_loss);
			printf("BEST LOSS: %f\n",best_loss);
			printf("LOSS ITER: %d\n",loss_iter);
		}

    outer_iter++;  
		spl_weight /= spl_factor;
  } // end outer loop
	fclose(fexamples);
	fclose(ftime);
	fclose(flatent);
  fclose(fslack);
  fclose(fentropy);
  fclose(fnovelty);
  fclose(fdifficulty);
  fclose(floss);
  // fclose(ffycache);
  free(slacks);
  free(entropies);
  free(novelties);
  free(difficulties);
  free(losses);
  

  /* write structural model */
  write_struct_model(modelfile, &sm, &sparm);
  // skip testing for the moment  

  /* free memory */
  free_struct_sample(alldata);
	if(ntrain < alldata.n)
	{
		free(sample.examples);
		free(val.examples);
	}
  free_struct_model(sm, &sparm);
  for(i=0;i<m;i++) {
    free_svector(fycache[i]);
  }
  free(fycache);

	free(valid_examples);
   
  return(0); 
  
}



void my_read_input_parameters(int argc, char *argv[], char *trainfile,char* modelfile, char *examplesfile, char *timefile, char *latentfile,char *slackfile, char *uncertaintyfile, char *noveltyfile, char *lossfile, char *fycachefile, char *difficultyfile, LEARN_PARM *learn_parm, KERNEL_PARM *kernel_parm, STRUCT_LEARN_PARM *struct_parm,double *init_spl_weight, double *spl_factor) {
  
  long i;
	char filestub[1024];

  /* set default */
  learn_parm->maxiter=20000;
  learn_parm->svm_maxqpsize=100;
  learn_parm->svm_c=100.0;
  learn_parm->eps=0.001;
  learn_parm->biased_hyperplane=12345; /* store random seed */
  learn_parm->remove_inconsistent=10; 
  kernel_parm->kernel_type=0;
  kernel_parm->rbf_gamma=0.05;
  kernel_parm->coef_lin=1;
  kernel_parm->coef_const=1;
  kernel_parm->poly_degree=3;
	/* default: no self-paced learning */
	*init_spl_weight = 0.0;
	*spl_factor = 1.3;
	struct_parm->optimizer_type = 0; /* default: cutting plane, change to 1 for stochastic subgradient descent*/
	struct_parm->init_valid_fraction = 0.5;
  struct_parm->uncertainty_weight = 0.0;
  struct_parm->novelty_weight = 0.0;
  struct_parm->renyi_exponent = -1.0; // see get_example_scores for what this means
  struct_parm->print_extensive = 0;
  struct_parm->reduced_size = 0;
  struct_parm->init_valid_fraction_pos = 0.0;
  struct_parm->margin_type = 0; // 0 means margin rescaling, 1 means opposite y
	struct_parm->using_argmax = 1; // 0 means use expectation, 1 means argmax

  struct_parm->custom_argc=0;
	/*-------------------------------------------------------------------------------*/

  for(i=1;(i<argc) && ((argv[i])[0] == '-');i++) {
    switch ((argv[i])[1]) {
    case 'a': i++; struct_parm->using_argmax=atoi(argv[i]); break;
    case 'c': i++; learn_parm->svm_c=atof(argv[i]); break;
    case 'd': i++; kernel_parm->poly_degree=atol(argv[i]); break;
    case 'e': i++; learn_parm->eps=atof(argv[i]); break;
		case 'f': i++; struct_parm->init_valid_fraction = atof(argv[i]); break;
    case 'g': i++; kernel_parm->rbf_gamma=atof(argv[i]); break;
		case 'k': i++; *init_spl_weight = atof(argv[i]); break;
		case 'm': i++; *spl_factor = atof(argv[i]); break;
    case 'n': i++; learn_parm->maxiter=atol(argv[i]); break;
		case 'o': i++; struct_parm->optimizer_type = atoi(argv[i]); break;
    case 'p': i++; learn_parm->remove_inconsistent=atol(argv[i]); break; 
    case 'r': i++; learn_parm->biased_hyperplane=atol(argv[i]); break; 
    case 's': i++; learn_parm->svm_maxqpsize=atol(argv[i]); break; 
    case 't': i++; kernel_parm->kernel_type=atol(argv[i]); break;
    case 'x': i++; struct_parm->renyi_exponent=atof(argv[i]); break;
    case 'z': i++; struct_parm->reduced_size = atoi(argv[i]); break;
    case '-': strcpy(struct_parm->custom_argv[struct_parm->custom_argc++],argv[i]);i++; strcpy(struct_parm->custom_argv[struct_parm->custom_argc++],argv[i]);break; 
    default: printf("\nUnrecognized option %s!\n\n",argv[i]);
      exit(0);
    }

  }
	*init_spl_weight = (*init_spl_weight)/learn_parm->svm_c;

  if(i>=argc) {
    printf("\nNot enough input parameters!\n\n");
    my_wait_any_key();
    exit(0);
  }
  strcpy (trainfile, argv[i]);

  if((i+1)<argc) {
    strcpy (modelfile, argv[i+1]);
  }
	else {
		strcpy (modelfile, "lssvm.model");
	}

	if((i+2)<argc) {
		strcpy (filestub, argv[i+2]);
	}
	else {
		strcpy (filestub, "lssvm");
	}

	sprintf(examplesfile,"%s.examples",filestub);
	sprintf(timefile,"%s.time",filestub);
	sprintf(latentfile,"%s.latent",filestub);
  sprintf(slackfile,"%s.slack",filestub);
  sprintf(uncertaintyfile,"%s.entropy",filestub);
  sprintf(noveltyfile,"%s.novelty",filestub);
  sprintf(lossfile,"%s.loss",filestub);
	sprintf(fycachefile,"%s.fycache",filestub);
	sprintf(difficultyfile,"%s.difficulty",filestub);

	/* self-paced learning weight should be non-negative */
	if(*init_spl_weight < 0.0)
		*init_spl_weight = 0.0;
	/* self-paced learning factor should be greater than 1.0 */
	if(*spl_factor < 1.0)
		*spl_factor = 1.1;

  parse_struct_parameters(struct_parm);
}


void my_wait_any_key()
{
  printf("\n(more)\n");
  (void)getc(stdin);
}

int resize_cleanup(int size_active, int **ptr_idle, double **ptr_alpha, double **ptr_delta, DOC ***ptr_dXc, 
		double ***ptr_G, int *mv_iter) {
  int i,j, new_size_active;
  long k;

  int *idle=*ptr_idle;
  double *alpha=*ptr_alpha;
  double *delta=*ptr_delta;
	DOC	**dXc = *ptr_dXc;
	double **G = *ptr_G;
	int new_mv_iter;

  i=0;
  while ((i<size_active)&&(idle[i]<IDLE_ITER)) i++;
  j=i;
  while((j<size_active)&&(idle[j]>=IDLE_ITER)) j++;

  while (j<size_active) {
    /* copying */
    alpha[i] = alpha[j];
    delta[i] = delta[j];
		free(G[i]);
		G[i] = G[j];
		G[j] = NULL;
    free_example(dXc[i],0);
    dXc[i] = dXc[j];
    dXc[j] = NULL;
		if(j == *mv_iter)
			new_mv_iter = i;

    i++;
    j++;
    while((j<size_active)&&(idle[j]>=IDLE_ITER)) j++;
  }
  for (k=i;k<size_active;k++) {
		if (G[k]!=NULL) free(G[k]);
    if (dXc[k]!=NULL) free_example(dXc[k],0);
  }
	*mv_iter = new_mv_iter;
  new_size_active = i;
  alpha = (double*)realloc(alpha, sizeof(double)*new_size_active);
  delta = (double*)realloc(delta, sizeof(double)*new_size_active);
	G = (double **) realloc(G, sizeof(double *)*new_size_active);
  dXc = (DOC**)realloc(dXc, sizeof(DOC*)*new_size_active);
  assert(dXc!=NULL);

  /* resize idle */
  i=0;
  while ((i<size_active)&&(idle[i]<IDLE_ITER)) i++;
  j=i;
  while((j<size_active)&&(idle[j]>=IDLE_ITER)) j++;

  while (j<size_active) {
    idle[i] = idle[j];
		for (k=0;k<new_size_active;k++) {
			G[k][i] = G[k][j];
		}
    i++;
    j++;
    while((j<size_active)&&(idle[j]>=IDLE_ITER)) j++;
  }  
  idle = (int*)realloc(idle, sizeof(int)*new_size_active);
	for (k=0;k<new_size_active;k++) {
		G[k] = (double*)realloc(G[k], sizeof(double)*new_size_active);
	}

  *ptr_idle = idle;
  *ptr_alpha = alpha;
  *ptr_delta = delta;
	*ptr_G = G;
  *ptr_dXc = dXc;

  return(new_size_active);
}

void approximate_to_psd(double **G, int size_active, double eps)
{
	int i,j,k;
	double *copy_G = malloc(size_active*size_active*sizeof(double));
	double *eig_vec = malloc(size_active*size_active*sizeof(double));
	double *eig_val = malloc(size_active*sizeof(double));

	for(i = 0; i < size_active; i++)
		for(j = 0; j < size_active; j++)
			copy_G[i*size_active+j] = G[i][j];

	Jacobi_Cyclic_Method(eig_val,eig_vec,copy_G,size_active);

	for(i = 0; i < size_active; i++)
		for(j = 0; j < size_active; j++)
		{
			copy_G[i*size_active+j] = MAX(eig_val[i],eps)*eig_vec[j*size_active+i];
		}

	for(i = 0; i < size_active; i++)
		for(j = 0; j < size_active; j++)
		{
			G[i][j] = 0.0;
			for(k = 0; k < size_active; k++)
			{
				G[i][j] += eig_vec[i*size_active+k]*copy_G[k*size_active+j];
			}
		}

	free(copy_G);
	free(eig_vec);
	free(eig_val);
}

void Jacobi_Cyclic_Method(double eigenvalues[], double *eigenvectors, double *A, int n)
{
   int row, i, j, k, m;
   double *pAk, *pAm, *p_r, *p_e;
   double threshold_norm;
   double threshold;
   double tan_phi, sin_phi, cos_phi, tan2_phi, sin2_phi, cos2_phi;
   double sin_2phi, cos_2phi, cot_2phi;
   double dum1;
   double dum2;
   double dum3;
   double r;
   double max;

                  // Take care of trivial cases

   if ( n < 1) return;
   if ( n == 1) {
      eigenvalues[0] = *A;
      *eigenvectors = 1.0;
      return;
   }

          // Initialize the eigenvalues to the identity matrix.

   for (p_e = eigenvectors, i = 0; i < n; i++)
      for (j = 0; j < n; p_e++, j++)
         if (i == j) *p_e = 1.0; else *p_e = 0.0;
  
            // Calculate the threshold and threshold_norm.
 
   for (threshold = 0.0, pAk = A, i = 0; i < ( n - 1 ); pAk += n, i++) 
      for (j = i + 1; j < n; j++) threshold += *(pAk + j) * *(pAk + j);
   threshold = sqrt(threshold + threshold);
   threshold_norm = threshold * DBL_EPSILON;
   max = threshold + 1.0;
   while (threshold > threshold_norm) {
      threshold /= 10.0;
      if (max < threshold) continue;
      max = 0.0;
      for (pAk = A, k = 0; k < (n-1); pAk += n, k++) {
         for (pAm = pAk + n, m = k + 1; m < n; pAm += n, m++) {
            if ( fabs(*(pAk + m)) < threshold ) continue;

                 // Calculate the sin and cos of the rotation angle which
                 // annihilates A[k][m].

            cot_2phi = 0.5 * ( *(pAk + k) - *(pAm + m) ) / *(pAk + m);
            dum1 = sqrt( cot_2phi * cot_2phi + 1.0);
            if (cot_2phi < 0.0) dum1 = -dum1;
            tan_phi = -cot_2phi + dum1;
            tan2_phi = tan_phi * tan_phi;
            sin2_phi = tan2_phi / (1.0 + tan2_phi);
            cos2_phi = 1.0 - sin2_phi;
            sin_phi = sqrt(sin2_phi);
            if (tan_phi < 0.0) sin_phi = - sin_phi;
            cos_phi = sqrt(cos2_phi); 
            sin_2phi = 2.0 * sin_phi * cos_phi;
            cos_2phi = cos2_phi - sin2_phi;

                     // Rotate columns k and m for both the matrix A 
                     //     and the matrix of eigenvectors.

            p_r = A;
            dum1 = *(pAk + k);
            dum2 = *(pAm + m);
            dum3 = *(pAk + m);
            *(pAk + k) = dum1 * cos2_phi + dum2 * sin2_phi + dum3 * sin_2phi;
            *(pAm + m) = dum1 * sin2_phi + dum2 * cos2_phi - dum3 * sin_2phi;
            *(pAk + m) = 0.0;
            *(pAm + k) = 0.0;
            for (i = 0; i < n; p_r += n, i++) {
               if ( (i == k) || (i == m) ) continue;
               if ( i < k ) dum1 = *(p_r + k); else dum1 = *(pAk + i);
               if ( i < m ) dum2 = *(p_r + m); else dum2 = *(pAm + i);
               dum3 = dum1 * cos_phi + dum2 * sin_phi;
               if ( i < k ) *(p_r + k) = dum3; else *(pAk + i) = dum3;
               dum3 = - dum1 * sin_phi + dum2 * cos_phi;
               if ( i < m ) *(p_r + m) = dum3; else *(pAm + i) = dum3;
            }
            for (p_e = eigenvectors, i = 0; i < n; p_e += n, i++) {
               dum1 = *(p_e + k);
               dum2 = *(p_e + m);
               *(p_e + k) = dum1 * cos_phi + dum2 * sin_phi;
               *(p_e + m) = - dum1 * sin_phi + dum2 * cos_phi;
            }
         }
         for (i = 0; i < n; i++)
            if ( i == k ) continue;
            else if ( max < fabs(*(pAk + i))) max = fabs(*(pAk + i));
      }
   }
   for (pAk = A, k = 0; k < n; pAk += n, k++) eigenvalues[k] = *(pAk + k); 
}
