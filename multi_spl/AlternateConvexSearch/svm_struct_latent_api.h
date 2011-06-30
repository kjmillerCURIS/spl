/************************************************************************/
/*                                                                      */
/*   svm_struct_latent_api.h                                            */
/*                                                                      */
/*   API function interface for Latent SVM^struct                       */
/*                                                                      */
/*   Author: Chun-Nam Yu                                                */
/*   Date: 21.Dec.08                                                    */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/

#include "./svm_light/svm_common.h"
#include "svm_struct_latent_api_types.h"
#include <float.h>

int get_sample_size(char * file);
SAMPLE read_struct_examples(char *file, STRUCTMODEL * sm, STRUCT_LEARN_PARM *sparm);
int get_num_bbox_positions(int image_length, int bbox_length, int bbox_step_length);
void read_kernel_info(char * kernel_info_file, STRUCTMODEL * sm);
void init_struct_model(int sample_size, char * kernel_info_file, STRUCTMODEL *sm);
void init_latent_variables(SAMPLE *sample, LEARN_PARM *lparm, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
int in_bounding_box(int pixel_x, int pixel_y, LATENT_VAR h, STRUCTMODEL * sm);
FILE * open_kernelized_image_file(PATTERN x, int kernel_ind, STRUCTMODEL * sm);
void fill_max_pool(PATTERN x, LATENT_VAR h, int kernel_ind, double * max_pool_segment, STRUCTMODEL * sm);
SVECTOR *psi(PATTERN x, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
void classify_struct_example(PATTERN x, LABEL *y, LATENT_VAR *h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
void find_most_violated_constraint_marginrescaling(PATTERN x, LATENT_VAR hstar, LABEL y, LABEL *ybar, LATENT_VAR *hbar, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
void find_most_violated_constraint_differenty(PATTERN x, LATENT_VAR hstar, LABEL y, LABEL *ybar, LATENT_VAR *hbar, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
void initialize_most_violated_constraint_search(PATTERN x, LATENT_VAR hstar, LABEL y, LABEL *ybar, LATENT_VAR *hbar, double * max_score, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
double compute_w_T_psi(PATTERN *x, int position_x, int position_y, int class, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
LATENT_VAR infer_latent_variables(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
double loss(LABEL y, LABEL ybar, LATENT_VAR hbar, STRUCT_LEARN_PARM *sparm);
void write_struct_model(char *file, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
void read_struct_model(char *model_file, STRUCTMODEL *sm);
void free_struct_model(STRUCTMODEL sm, STRUCT_LEARN_PARM *sparm);
void free_pattern(PATTERN x);
void free_label(LABEL y);
void free_latent_var(LATENT_VAR h);
void free_struct_sample(SAMPLE s);
void parse_struct_parameters(STRUCT_LEARN_PARM *sparm);

void print_latent_var(LATENT_VAR h, FILE *flatent);
void print_label(LABEL l, FILE *flabel);


