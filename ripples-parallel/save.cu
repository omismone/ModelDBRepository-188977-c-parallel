#include "omislib.cuh"

#define RELPATH "C:\\Users\\mclab\\Desktop\\simone\\thesis\\ripples-parallel\\ripples-parallel\\scripts\\data\\"
#define BUF 1024

void save(struct Veg* veg, struct vec* lfp, struct Tsp* tspE, struct Tsp* tspI, struct Inp* inp, struct inpseq* inps, int T, int NE, int NI) {

    printf("\nsaving results..\n");

    char filename[BUF];
    char tmp[BUF];
    FILE* fp;

    /* veg */
    //veg.ne
    strcpy(filename, RELPATH);
    strcat(filename, "veg_ne.txt");
    fp = fopen(filename, "w");
    fprintf(fp, "%d", veg->ne);
    fclose(fp);
    //veg.ni
    strcpy(filename, RELPATH);
    strcat(filename, "veg_ni.txt");
    fp = fopen(filename, "w");
    fprintf(fp, "%d", veg->ni);
    fclose(fp);
    //veg.E
    strcpy(filename, RELPATH);
    strcat(filename, "veg_E.txt");
    fp = fopen(filename, "w");
    for (int i = 0; i < lfp->size; i++) {
        fprintf(fp, "%lf ", veg->E[i]);
    }
    fclose(fp);
    //veg.I
    strcpy(filename, RELPATH);
    strcat(filename, "veg_I.txt");
    fp = fopen(filename, "w");
    for (int i = 0; i < lfp->size; i++) {
        fprintf(fp, "%lf ", veg->I[i]);
    }
    fclose(fp);


    /* tspI */
    //tspI.times
    strcpy(filename, RELPATH);
    strcat(filename, "tsp_I_times.txt");
    fp = fopen(filename, "w");
    for (int i = 0; i < tspI->times.size; i++) {
        fprintf(fp, "%lf ", tspI->times.val[i]);
    }
    fclose(fp);
    //tspI.celln
    strcpy(filename, RELPATH);
    strcat(filename, "tsp_I_celln.txt");
    fp = fopen(filename, "w");
    for (int i = 0; i < tspI->celln.size; i++) {
        fprintf(fp, "%lf ", tspI->celln.val[i]);
    }
    fclose(fp);


    /* tspE */
    //tspE.times
    strcpy(filename, RELPATH);
    strcat(filename, "tsp_E_times.txt");
    fp = fopen(filename, "w");
    for (int i = 0; i < tspE->times.size; i++) {
        fprintf(fp, "%lf ", tspE->times.val[i]);
    }
    fclose(fp);
    //tspE.celln
    strcpy(filename, RELPATH);
    strcat(filename, "tsp_E_celln.txt");
    fp = fopen(filename, "w");
    for (int i = 0; i < tspE->celln.size; i++) {
        fprintf(fp, "%lf ", tspE->celln.val[i]);
    }
    fclose(fp);


    /* T */
    strcpy(filename, RELPATH);
    strcat(filename, "T.txt");
    fp = fopen(filename, "w");
    fprintf(fp, "%d", T);
    fclose(fp);


    /* NI */
    strcpy(filename, RELPATH);
    strcat(filename, "NI.txt");
    fp = fopen(filename, "w");
    fprintf(fp, "%d", NI);
    fclose(fp);


    /* NE */
    strcpy(filename, RELPATH);
    strcat(filename, "NE.txt");
    fp = fopen(filename, "w");
    fprintf(fp, "%d", NE);
    fclose(fp);


    /* lfp */
    strcpy(filename, RELPATH);
    strcat(filename, "lfp.txt");
    fp = fopen(filename, "w");
    for (int i = 0; i < lfp->size; i++) {
        fprintf(fp, "%lf ", lfp->val[i]);
    }
    fclose(fp);


    /* inpseq */
    //inpseq.slp
    strcpy(filename, RELPATH);
    strcat(filename, "inpseq_slp.txt");
    fp = fopen(filename, "w");
    fprintf(fp, "%f", inps->slp);
    fclose(fp);
    //inpseq.on
    strcpy(filename, RELPATH);
    strcat(filename, "inpseq_on.txt");
    fp = fopen(filename, "w");
    for (int i = 0; i < inps->on.size; i++) {
        fprintf(fp, "%lf ", inps->on.val[i]);
    }
    fclose(fp);
    //inpseq.length
    strcpy(filename, RELPATH);
    strcat(filename, "inpseq_length.txt");
    fp = fopen(filename, "w");
    fprintf(fp, "%f", inps->length);
    fclose(fp);


    /* inp */
    //inp.Edc
    strcpy(filename, RELPATH);
    strcat(filename, "inp_Edc.txt");
    fp = fopen(filename, "w");
    for (int i = 0; i < NE; i++) {
        fprintf(fp, "%lf ", inp->Edc[i]);
    }
    fclose(fp);
    //inp.Idc
    strcpy(filename, RELPATH);
    strcat(filename, "inp_Idc.txt");
    fp = fopen(filename, "w");
    for (int i = 0; i < NI; i++) {
        fprintf(fp, "%lf ", inp->Idc[i]);
    }
    fclose(fp);
    //inp.Etrace
    strcpy(filename, RELPATH);
    strcat(filename, "inp_Etrace.txt");
    fp = fopen(filename, "w");
    for (int i = 0; i < inp->Etrace.size[0]; i++) {
        for (int j = 0; j < inp->Etrace.size[1]; j++) {
            fprintf(fp, "%lf ", inp->Etrace.val[i * inp->Etrace.size[1] + j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    //inp.Itrace
    strcpy(filename, RELPATH);
    strcat(filename, "inp_Itrace.txt");
    fp = fopen(filename, "w");
    for (int i = 0; i < inp->Itrace.size[0]; i++) {
        for (int j = 0; j < inp->Itrace.size[1]; j++) {
            fprintf(fp, "%lf ", inp->Itrace.val[i * inp->Itrace.size[1] + j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);

    return;
}
