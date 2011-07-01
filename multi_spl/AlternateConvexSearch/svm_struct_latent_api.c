/************************************************************************/
/*                                                                      */
/*   svm_struct_latent_api.c                                            */
/*                                                                      */
/*   API function definitions for Latent SVM^struct                     */
/*                                                                      */
/*   Author: Chun-Nam Yu                                                */
/*   Date: 17.Dec.08                                                    */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include "svm_struct_latent_api_types.h"
#include "./SFMT-src-1.3.3/SFMT.h"

#define MAX_INPUT_LINE_LENGTH 10000
#define DELTA 1

int get_sample_size(char * file) {
  int sample_size;
  FILE * fp = fopen(file, "r");
  fscanf(fp, "%d\n", &sample_size);
  fclose(fp);
  return sample_size;
}

IMAGE_KERNEL_CACHE ** init_cached_images(STRUCTMODEL * sm) {
  return (IMAGE_KERNEL_CACHE **)calloc(sm->n, sizeof(IMAGE_KERNEL_CACHE *));
}

void free_cached_images(IMAGE_KERNEL_CACHE ** cached_images, STRUCTMODEL * sm) {
  int i, k;
  for (i = 0; i < sm->n; ++i) {
    if (cached_images[i] != NULL) {
      for (k = 0; k < sm->num_kernels; ++k) {
	free(cached_images[i][k].points_and_descriptors);
	free(cached_images[i][k].begins);
	free(cached_images[i][k].ends);
      }
    }
    free(cached_images[i]);
  }
  free(cached_images);
}

SAMPLE read_struct_examples(char *file, STRUCTMODEL * sm, STRUCT_LEARN_PARM *sparm) {
  /*
    Gets and stores image file name, line number (i.e. index), label, width, and height for each example.
    Width and height should be in units such that width * height = number of options for h.
  */

  SAMPLE sample;
  int num_examples,label,height,width;
	int i,j,k,l;
  FILE *fp;
  char line[MAX_INPUT_LINE_LENGTH]; 
  char *pchar, *last_pchar;

  fp = fopen(file,"r");
  if (fp==NULL) {
    printf("Cannot open input file %s!\n", file);
	exit(1);
  }
  fgets(line, MAX_INPUT_LINE_LENGTH, fp);
  num_examples = atoi(line);
  sample.n = num_examples;
  sample.examples = (EXAMPLE*)malloc(sizeof(EXAMPLE)*num_examples);
  
  for (i=0;(!feof(fp))&&(i<num_examples);i++) {
    fgets(line, MAX_INPUT_LINE_LENGTH, fp);

    pchar = line;
    while ((*pchar)!=' ') pchar++;
    *pchar = '\0';
    strcpy(sample.examples[i].x.image_path, line);
    pchar++;

    /* label: {0, 1} */
    last_pchar = pchar;
    while ((*pchar)!=' ') pchar++;
    *pchar = '\0';
    label = atoi(last_pchar);
    pchar++;

    last_pchar = pchar;
    while ((*pchar)!=' ') pchar++;
    *pchar = '\0';
    height = atoi(last_pchar);
    pchar++;

    last_pchar = pchar;
    while ((*pchar)!='\n') pchar++;
    *pchar = '\0';
    width = atoi(last_pchar);

    assert(label >= 0 && label < sparm->n_classes);
    sample.examples[i].y.label = label;
    sample.examples[i].x.width = get_num_bbox_positions(width, sm->bbox_width, sm->bbox_step_x);
    sample.examples[i].x.height = get_num_bbox_positions(height, sm->bbox_height, sm->bbox_step_y);
    sample.examples[i].x.example_id = i;
    sample.examples[i].x.example_cost = 1.0;
  }
  assert(i==num_examples);
  fclose(fp);  
  return(sample); 
}

int get_num_bbox_positions(int image_length, int bbox_length, int bbox_step_length) {
  return (int)ceil((1.0 * image_length - 1.0 * bbox_length) / (1.0 * bbox_step_length));
}

//file format is "<number of kernels>\n<kernel 0 name>\n<kernel 0 size>\n<kernel 1 name>\n...."
void read_kernel_info(char * kernel_info_file, STRUCTMODEL * sm) {
  int k;
  FILE * fp = fopen(kernel_info_file, "r");
  fscanf(fp, "%d\n", &(sm->num_kernels));
  sm->kernel_names = (char**)malloc(sm->num_kernels * sizeof(char*));
  sm->kernel_sizes = (int*)malloc(sm->num_kernels * sizeof(int));
  char cur_kernel_name[1024]; //if you need more than 1023 characters to name a kernel, you need help
  for (k = 0; k < sm->num_kernels; ++k) {
    fscanf(fp, "%s\n", cur_kernel_name);
    sm->kernel_names[k] = strdup(cur_kernel_name);
    fscanf(fp, "%d\n", &(sm->kernel_sizes[k]));
  }
  sm->sizePsi = 0;
  for (k = 0; k < sm->num_kernels; ++k) {
    sm->sizePsi += sm->kernel_sizes[k];
  }
}

void init_struct_model(int sample_size, char * kernel_info_file, STRUCTMODEL *sm) {
/*
  Initialize parameters in STRUCTMODEL sm. Set the dimension 
  of the feature space sm->sizePsi. Can also initialize your own
  variables in sm here. 
*/

  int i,j,k;

  read_kernel_info(kernel_info_file, sm);

  sm->n = sample_size;
}

void init_latent_variables(SAMPLE *sample, LEARN_PARM *lparm, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Initialize latent variables in the first iteration of training.
  Latent variables are stored at sample.examples[i].h, for 1<=i<=sample.n.
*/
	
  int i;
  /* initialize the RNG */
	init_gen_rand(sparm->rng_seed);

	for (i=0;i<sample->n;i++) {
		sample->examples[i].h.position_x = (long) floor(genrand_res53()*(sample->examples[i].x.width-1));
		sample->examples[i].h.position_y = (long) floor(genrand_res53()*(sample->examples[i].x.height-1));
		if(sample->examples[i].h.position_x < 0 || sample->examples[i].h.position_x >= sample->examples[i].x.width-1)
			sample->examples[i].h.position_x = (long) 0;
		if(sample->examples[i].h.position_y < 0 || sample->examples[i].h.position_y >= sample->examples[i].x.height-1)
			sample->examples[i].h.position_y = (long) 0;
	}
}

int in_bounding_box(int pixel_x, int pixel_y, LATENT_VAR h, STRUCTMODEL * sm) {
  int bbox_start_x = h.position_x * sm->bbox_step_x;
  int bbox_start_y = h.position_y * sm->bbox_step_y;
  int bbox_end_x = bbox_start_x + sm->bbox_width;
  int bbox_end_y = bbox_start_y + sm->bbox_height;
  return (pixel_x >= bbox_start_x) && (pixel_y >= bbox_start_y) && (pixel_x < bbox_end_x) && (pixel_y < bbox_end_y);
}

//if the contents of files are ever cached, this would be a good place to implement that cacheing
FILE * open_kernelized_image_file(PATTERN x, int kernel_ind, STRUCTMODEL * sm) {
  char file_path[1024];
  strcpy(file_path, x.image_path);
  strcat(file_path, "/");
  strcat(file_path, sm->kernel_names[kernel_ind]);
  return fopen(file_path, "r");
}

void fill_image_kernel_cache(PATTERN x, int kernel_ind, IMAGE_KERNEL_CACHE * ikc, STRUCTMODEL * sm) {
  int p, l;
  char throwaway_line[1024];
  FILE * fp = open_kernelized_image_file(x, kernel_ind, sm);
  fscanf(fp, "%d\n", &(ikc->num_points));
  ikc->points_and_descriptors = (POINT_AND_DESCRIPTOR *)calloc(ikc->num_points, sizeof(POINT_AND_DESCRIPTOR));
  fscanf(fp, "%s\n", throwaway_line);
  for (p = 0; p < ikc->num_points; ++p) {
    fscanf(fp, "(%d,%d):%d\n", &(ikc->points_and_descriptors[p].y), &(ikc->points_and_descriptors[p].x), &(ikc->points_and_descriptors[p].descriptor));
  }
  fclose(fp);
  ikc->begins = (int *)calloc(x.width * x.height, sizeof(int));
  ikc->ends = (int *)malloc(x.width * x.height * sizeof(int));
  for (l = 0; l < x.width * x.height; ++l) {
    ikc->ends[l] = ikc->num_points;
  }
}

void try_cache_image(PATTERN x, IMAGE_KERNEL_CACHE ** cached_images, STRUCTMODEL * sm) {
  int k;
  if (cached_images[x.example_id] == NULL) {
    printf("$"); fflush(stdout);
    cached_images[x.example_id] = (IMAGE_KERNEL_CACHE *)malloc(sm->num_kernels * sizeof(IMAGE_KERNEL_CACHE));
    IMAGE_KERNEL_CACHE * kernel_caches_for_image = cached_images[x.example_id];
    for (k = 0; k < sm->num_kernels; ++k) {
      fill_image_kernel_cache(x, k, &(kernel_caches_for_image[k]), sm);
    }
  }
}

void fill_max_pool(PATTERN x, LATENT_VAR h, int kernel_ind, double * max_pool_segment, IMAGE_KERNEL_CACHE ** cached_images, STRUCTMODEL * sm){
  int p, d;

  clock_t start_time, finish_time;

  start_time = clock();

  int updated_begin, updated_end; //the "update" really only happens the first time you call this function, but treating things this way won't waste any significant amount of time

  updated_begin = -1;
  updated_end = 0;

  int current_begin = cached_images[x.example_id][kernel_ind].begins[h.position_x * x.height + h.position_y];
  int current_end = cached_images[x.example_id][kernel_ind].ends[h.position_x * x.height + h.position_y];
  if (current_end == 0) { //this means that we know from a previous update that no points are in this bounding box
    return;
  }
  

  for (p = current_begin; p < current_end; ++p) {
    int point_x = cached_images[x.example_id][kernel_ind].points_and_descriptors[p].x;
    int point_y = cached_images[x.example_id][kernel_ind].points_and_descriptors[p].y;
    int descriptor = cached_images[x.example_id][kernel_ind].points_and_descriptors[p].descriptor;
    if (in_bounding_box(point_x, point_y, h, sm)) {
      max_pool_segment[descriptor - 1] = 1.0;
      if (updated_begin == -1) {
	updated_begin = p;
      }
      updated_end = p + 1;
    }
  }
  
  cached_images[x.example_id][kernel_ind].begins[h.position_x * x.height + h.position_y] = updated_begin;
  cached_images[x.example_id][kernel_ind].ends[h.position_x * x.height + h.position_y] = updated_end;

  finish_time = clock();

  //printf("fill_max_pool() took %f ticks to iterate through all points (according to clock()).\n", ((double) (finish_time - start_time)));

  double sum = 0.0;
    for (d = 0; d < sm->kernel_sizes[kernel_ind]; ++d) {
      sum += max_pool_segment[d];
    }
  for (d = 0; d < sm->kernel_sizes[kernel_ind]; ++d) {
    max_pool_segment[d] /= sum;
  }
}

SVECTOR *psi(PATTERN x, LABEL y, LATENT_VAR h, IMAGE_KERNEL_CACHE ** cached_images, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Creates the feature vector \Psi(x,y,h) and return a pointer to 
  sparse vector SVECTOR in SVM^light format. The dimension of the 
  feature vector returned has to agree with the dimension in sm->sizePsi. 
*/
  struct timeval start_time;
  struct timeval finish_time;
  gettimeofday(&start_time, NULL);

  try_cache_image(x, cached_images, sm);

  SVECTOR * fvec = NULL;
  double * max_pool = (double *)calloc(sm->sizePsi + 1, sizeof(double));
  //binary labelling for now - 1 means there's a car, 0 means there's no car
  if (y.label) {
    int k;
    int start_ind = 1;
    for (k = 0; k < sm->num_kernels; ++k) {
      fill_max_pool(x, h, k, &(max_pool[start_ind]), cached_images, sm);
      start_ind += sm->kernel_sizes[k];
    }
  }
  
  gettimeofday(&finish_time, NULL);

  if (y.label) {
    int million = 1000000;
    int microseconds = million * (int)(finish_time.tv_sec - start_time.tv_sec) + (int)(finish_time.tv_usec - start_time.tv_usec);
    //    printf("psi() took %d microseconds.\n", microseconds);
  }
  
  fvec = create_svector_n(max_pool, sm->sizePsi, "", 1);
  free(max_pool);
  return fvec;
}

double compute_w_T_psi(PATTERN *x, int position_x, int position_y, int class, IMAGE_KERNEL_CACHE ** cached_images, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
  double w_T_psi;
  LABEL y;
  LATENT_VAR h;
  y.label = class;
  h.position_x = position_x;
  h.position_y = position_y;
  SVECTOR * psi_vect = psi(*x, y, h, cached_images, sm, sparm);
  w_T_psi = sprod_ns(sm->w, psi_vect);
  free_svector(psi_vect);
  return w_T_psi;
}


void classify_struct_example(PATTERN x, LABEL *y, LATENT_VAR *h, IMAGE_KERNEL_CACHE ** cached_images, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Makes prediction with input pattern x with weight vector in sm->w,
  i.e., computing argmax_{(y,h)} <w,psi(x,y,h)>. 
  Output pair (y,h) are stored at location pointed to by 
  pointers *y and *h. 
*/
  
	int i;
	int width = x.width;
	int height = x.height;
	int cur_class, cur_position_x, cur_position_y;
	double max_score;
	double score;
	FILE	*fp;

	max_score = -DBL_MAX;
	for(cur_position_x = 0; cur_position_x < width; cur_position_x++) {
		for(cur_position_y = 0; cur_position_y < height; cur_position_y++) {
			for(cur_class = 0; cur_class < sparm->n_classes; cur_class++) {
			  score = compute_w_T_psi(&x, h->position_x, h->position_y, y->label, cached_images, sm, sparm);
				if(score > max_score) {
					max_score = score;
					y->label = cur_class;
					h->position_x = cur_position_x;
					h->position_y = cur_position_y;
				}
			}
		}
	}

	return;
}

void initialize_most_violated_constraint_search(PATTERN x, LATENT_VAR hstar, LABEL y, LABEL *ybar, LATENT_VAR *hbar, double * max_score, IMAGE_KERNEL_CACHE ** cached_images, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
  hbar->position_x = hstar.position_x;
  hbar->position_y = hstar.position_y;
  ybar->label = y.label;
  *max_score = compute_w_T_psi(&x, hbar->position_x, hbar->position_y, ybar->label, cached_images, sm, sparm);
}

void find_most_violated_constraint_marginrescaling(PATTERN x, LATENT_VAR hstar, LABEL y, LABEL *ybar, LATENT_VAR *hbar, IMAGE_KERNEL_CACHE ** cached_images, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Finds the most violated constraint (loss-augmented inference), i.e.,
  computing argmax_{(ybar,hbar)} [<w,psi(x,ybar,hbar)> + loss(y,ybar,hbar)].
  The output (ybar,hbar) are stored at location pointed by 
  pointers *ybar and *hbar. 
*/
  printf("width = %d, height = %d\n", x.width, x.height);

  time_t start_time = time(NULL);
	int width = x.width;
	int height = x.height;
	int cur_class, cur_position_x, cur_position_y;
	double max_score,score;
	FILE	*fp;
	
	//make explicit the idea that (y, hstar) is what's returned if the constraint is not violated
	initialize_most_violated_constraint_search(x, hstar, y, ybar, hbar, &max_score, cached_images, sm, sparm);
	
	for(cur_position_x = 0; cur_position_x < width; cur_position_x++) {
		for(cur_position_y = 0; cur_position_y < height; cur_position_y++) {
			for(cur_class = 0; cur_class < sparm->n_classes; cur_class++) {
			        score = compute_w_T_psi(&x, cur_position_x, cur_position_y, cur_class, cached_images, sm, sparm);
				if(cur_class != y.label)
					score += 1;
				if(score > max_score) {
					max_score = score;
					ybar->label = cur_class;
					hbar->position_x = cur_position_x;
					hbar->position_y = cur_position_y;
				}
			}
		}
	}

	time_t finish_time = time(NULL);
	printf("find_most_violated_constraint_marginrescaling took %d seconds to do %d h values.\n", (int)finish_time - (int)start_time, x.width * x.height);
	return;

}

void find_most_violated_constraint_differenty(PATTERN x, LATENT_VAR hstar, LABEL y, LABEL *ybar, LATENT_VAR *hbar, IMAGE_KERNEL_CACHE ** cached_images, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
  int width = x.width;
  int height = x.height;
  int cur_class, cur_position_x, cur_position_y;
  double max_score,score;
  FILE    *fp;

  //make explicit the idea that (y, hstar) is what's returned if the constraint is not violated
  initialize_most_violated_constraint_search(x, hstar, y, ybar, hbar, &max_score, cached_images, sm, sparm);

  for(cur_position_x = 0; cur_position_x < width; cur_position_x++) {
    for(cur_position_y = 0; cur_position_y < height; cur_position_y++) {
      for(cur_class = 0; cur_class < sparm->n_classes; cur_class++) {
	if (cur_class != y.label) {
	  score = DELTA + compute_w_T_psi(&x, cur_position_x, cur_position_y, cur_class, cached_images, sm, sparm);
	  if (score > max_score) {
	    max_score = score;
	    ybar->label = cur_class;
	    hbar->position_x = cur_position_x;
	    hbar->position_y = cur_position_y;
	  }
	}
      }
    }
  }

  return;

}


LATENT_VAR infer_latent_variables(PATTERN x, LABEL y, IMAGE_KERNEL_CACHE ** cached_images, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Complete the latent variable h for labeled examples, i.e.,
  computing argmax_{h} <w,psi(x,y,h)>. 
*/

  printf("width = %d, height = %d\n", x.width, x.height);
  time_t start_time = time(NULL);

  LATENT_VAR h;

  if (y.label == 0) {
    h.position_x = 0;
    h.position_y = 0;
    return h;
  }

	int i;
	int width = x.width;
	int height = x.height;
	int cur_position_x, cur_position_y;
	double max_score, score;
	FILE	*fp;

	max_score = -DBL_MAX;
	for(cur_position_x = 0; cur_position_x < width; cur_position_x++) {
		for(cur_position_y = 0; cur_position_y < height; cur_position_y++) {
		        score = compute_w_T_psi(&x, cur_position_x, cur_position_y, y.label, cached_images, sm, sparm);
			if(score > max_score) {
				max_score = score;
				h.position_x = cur_position_x;
				h.position_y = cur_position_y;
			}
		}
	}
  
	time_t finish_time = time(NULL);

	printf("infer_latent_variables() took %d seconds to do %d h values.\n", (int)finish_time - (int)start_time, x.width * x.height);

  return(h); 
}


double loss(LABEL y, LABEL ybar, LATENT_VAR hbar, STRUCT_LEARN_PARM *sparm) {
/*
  Computes the loss of prediction (ybar,hbar) against the
  correct label y. 
*/ 
  if (y.label==ybar.label) {
    return(0);
  } else {
    return(1);
  }
}

void write_struct_model(char *file, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Writes the learned weight vector sm->w to file after training. 
  Also writes bounding-box info (before sm->w)
*/
  FILE *modelfl;
  int i;
  
  modelfl = fopen(file,"w");
  if (modelfl==NULL) {
    printf("Cannot open model file %s for output!", file);
		exit(1);
  }
  
  fprintf(modelfl, "%d\n", sm->bbox_height);
  fprintf(modelfl, "%d\n", sm->bbox_width);
  fprintf(modelfl, "%d\n", sm->bbox_step_y);
  fprintf(modelfl, "%d\n", sm->bbox_step_x);

  for (i=1;i<sm->sizePsi+1;i++) {
    fprintf(modelfl, "%d:%.16g\n", i, sm->w[i]);
  }
  fclose(modelfl);
 
}

void read_struct_model(char *model_file, STRUCTMODEL * sm) {
/*
  Reads in the learned model parameters from file into STRUCTMODEL sm.
  The input file format has to agree with the format in write_struct_model().
*/

  FILE *modelfl;
  int i, fnum;
  double fweight;
  char line[1000];
  
  modelfl = fopen(model_file,"r");
  if (modelfl==NULL) {
    printf("Cannot open model file %s for input!", model_file);
	exit(1);
  }
  
  sm->w = (double*)calloc(sm->sizePsi + 1, sizeof(double));
  
  fscanf(modelfl, "%d\n", &(sm->bbox_height));
  fscanf(modelfl, "%d\n", &(sm->bbox_width));
  fscanf(modelfl, "%d\n", &(sm->bbox_step_y));
  fscanf(modelfl, "%d\n", &(sm->bbox_step_x));

  while (!feof(modelfl)) {
    fscanf(modelfl, "%d:%lf", &fnum, &fweight);
		sm->w[fnum] = fweight;
  }

  fclose(modelfl);
}

void free_struct_model(STRUCTMODEL sm, STRUCT_LEARN_PARM *sparm) {
/*
  Free any memory malloc'ed in STRUCTMODEL sm after training. 
*/
  int i, k;
  
  free(sm.w);

  for (k = 0; k < sm.num_kernels; ++k) {
    free(sm.kernel_names[k]);
  }
  free(sm.kernel_sizes);
  free(sm.kernel_names);
}

void free_pattern(PATTERN x) {
/*
  Free any memory malloc'ed when creating pattern x. 
*/

}

void free_label(LABEL y) {
/*
  Free any memory malloc'ed when creating label y. 
*/

} 

void free_latent_var(LATENT_VAR h) {
/*
  Free any memory malloc'ed when creating latent variable h. 
*/

}

void free_struct_sample(SAMPLE s) {
/*
  Free the whole training sample. 
*/
  int i;
  for (i=0;i<s.n;i++) {
    free_pattern(s.examples[i].x);
    free_label(s.examples[i].y);
    free_latent_var(s.examples[i].h);
  }
  free(s.examples);

}

void parse_struct_parameters(STRUCT_LEARN_PARM *sparm) {
/*
  Parse parameters for structured output learning passed 
  via the command line. 
*/
  int i;
  
  /* set default */
  sparm->rng_seed = 0;
  sparm->n_classes = 6;
  
  for (i=0;(i<sparm->custom_argc)&&((sparm->custom_argv[i])[0]=='-');i++) {
    switch ((sparm->custom_argv[i])[2]) {
      /* your code here */
      case 's': i++; sparm->rng_seed = atoi(sparm->custom_argv[i]); break;
      case 'n': i++; sparm->n_classes = atoi(sparm->custom_argv[i]); break;
      case 't': i++; sparm->margin_type = atoi(sparm->custom_argv[i]); break;
      default: printf("\nUnrecognized option %s!\n\n", sparm->custom_argv[i]); exit(0);
    }
  }
}

void copy_label(LABEL l1, LABEL *l2)
{
	l2->label = l1.label;
}

void copy_latent_var(LATENT_VAR lv1, LATENT_VAR *lv2)
{
	lv2->position_x = lv1.position_x;
	lv2->position_y = lv1.position_y;
}

void print_latent_var(LATENT_VAR h, FILE *flatent)
{
	fprintf(flatent,"%d %d ",h.position_x,h.position_y);
	fflush(flatent);
}

void print_label(LABEL l, FILE	*flabel)
{
	fprintf(flabel,"%d ",l.label);
	fflush(flabel);
}
