#include "omislib.cuh"

#define ENOISE_BIN_PATH1 "C:\\Users\\mclab\\Desktop\\simone\\thesis\\ripples-parallel\\ripples-parallel\\bin\\Enoise1.bin"
#define INOISE_BIN_PATH1 "C:\\Users\\mclab\\Desktop\\simone\\thesis\\ripples-parallel\\ripples-parallel\\bin\\Inoise1.bin"
#define ENOISE_BIN_PATH2 "C:\\Users\\mclab\\Desktop\\simone\\thesis\\ripples-parallel\\ripples-parallel\\bin\\Enoise2.bin"
#define INOISE_BIN_PATH2 "C:\\Users\\mclab\\Desktop\\simone\\thesis\\ripples-parallel\\ripples-parallel\\bin\\Inoise2.bin"


#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/*
*   @see https://en.wikipedia.org/wiki/Normal_distribution#Generating_values_from_normal_distribution
*/
double RandNormal()
{
    double x = 0, y = 0;
    while (x == 0 || y == 0) {
        x = (double)rand() / (double)RAND_MAX;
        y = (double)rand() / (double)RAND_MAX;
    }
    double z = sqrt(-2 * log(x)) * cos(2 * PI * y);

    return z;
}

/*
*   compare function (descending order) for quick sort
*/
static int AscendCmp(const void* a, const void* b) {
    double c = (*(double*)b - *(double*)a);
    if (c > 0)
        return -1;
    else if (c < 0)
        return 1;
    else
        return 0;
}

/*
*   compare function (ascending order) for quick sort
*/
static int DescendCmp(const void* a, const void* b) {
    double c = (*(double*)b - *(double*)a);
    if (c < 0)
        return -1;
    else if (c > 0)
        return 1;
    else
        return 0;
}

/*
*   Calculate the transposed of a matrix.
*
*   @param mat The matrix.
*/
void transpose(struct matrix* mat) {
    struct matrix res;
    res.size[0] = mat->size[1];
    res.size[1] = mat->size[0];
    res.val = (double*)malloc(res.size[0] * res.size[1] * sizeof(double));
    for (int i = 0; i < mat->size[0]; i++) {
        for (int j = 0; j < mat->size[1]; j++) {
            res.val[j * mat->size[0] + i] = mat->val[i * mat->size[1] + j];
        }
    }
    *mat = res;
}

__global__ void kernelGen(int NE, int NI, double* dev_sE, double* dev_sIE, double* dev_sI, double* dev_sEI, double* dev_GEE, double* dev_GIE, double* dev_GII, double* dev_GEI, double* dev_tmp1, double* dev_tmp2, double* dev_tmp3, double* dev_tmp4, int idt, int seqN, double* dev_vE, double* dev_vI, double* dev_lfp, double* dev_vbarE, double* dev_vbarI, double* dev_vegE, double* dev_vegI, int Eeg, int Ieg, int opt_storecurrs, double* dev_isynbarEtoE, double* dev_isynbarItoE, int lfp_size, double pVrevE, double pVrevI, double pglE, double pElE, double pslpE, double pVtE, double pCE, double paE, double ptwE, double pVrE, double pbE, double pglI, double pElI, double pslpI, double pVtI, double pCI, double paI, double ptwI, double pVrI, double pbI, double* dev_wE, double* dev_wI, double* dev_Enoise, double* dev_Inoise, int dev_Enoise_size_1, int dev_Inoise_size_1, double dev_dt, double* dev_erE, double* dev_edE, double* dev_erEI, double* dev_edEI, double* dev_erI, double* dev_edI, double* dev_erIE, double* dev_edIE, double dev_fdE, double dev_frE, double dev_fdEI, double dev_frEI, double dev_pvsE, double dev_pvsEI, double dev_fdI, double dev_frI, double dev_fdIE, double dev_frIE, double dev_pvsI, double dev_pvsIE, double* dev_tspEtimes, double* dev_tspEcelln, double* dev_tspItimes, double* dev_tspIcelln, int* dev_tspE_count, int* dev_tspI_count, double* dev_t) {
    int dev_ids;
    if (idt % 1000 == 1 && idt >= 1) {
        dev_ids = (idt - 1) / 1000 + 1 + 1000 * (seqN - 1);
        double dev_vE_sum = 0, dev_vI_sum = 0;
        for (int i = 0; i < NE; i++) {
            if (dev_vE[i] > 0)
                dev_vE[i] = 50;
            dev_vE_sum += dev_vE[i];
        }
        for (int i = 0; i < NI; i++) {
            if (dev_vI[i] > 0)
                dev_vI[i] = 50;
            dev_vI_sum += dev_vI[i];
        }
        dev_vbarE[dev_ids] = dev_vE_sum / NE;
        dev_vbarI[dev_ids] = dev_vI_sum / NI;
        dev_vegE[dev_ids] = dev_vE[Eeg];
        dev_vegI[dev_ids] = dev_vI[Ieg];
        dev_lfp[dev_ids] = 0;
        for (int i = 0; i < NE; i++) {
            dev_lfp[dev_ids] += (dev_tmp1[i] * (dev_vE[i] - pVrevE)) + (dev_tmp2[i] * (dev_vE[i] - pVrevI));
        }
        dev_lfp[dev_ids] /= NE;

        if (opt_storecurrs) {
            for (int i = 0; i < NE; i++) {
                dev_isynbarEtoE[i * lfp_size + dev_ids] = dev_tmp1[i] * (dev_vE[i] - pVrevE);
                dev_isynbarItoE[i * lfp_size + dev_ids] = dev_tmp2[i] * (dev_vE[i] - pVrevI);
            }
        }

    }

    for (int i = 0; i < NE; i++) {
        if (dev_vE[i] >= 0) {
            dev_tspEtimes[*dev_tspE_count] = (dev_t[(idt - 1)] / 1000) + seqN - 1;
            dev_tspEcelln[*dev_tspE_count] = i;
            (*dev_tspE_count)++;
        }
    }   
    for (int i = 0; i < NI; i++) {
        if (dev_vI[i] >= 0) {
            dev_tspItimes[*dev_tspI_count] = (dev_t[(idt - 1)] / 1000) + seqN - 1;
            dev_tspIcelln[*dev_tspI_count] = i;
            (*dev_tspI_count)++;
        }
    }

}

__global__ void kernelE(int NE, int NI, double* dev_sE, double* dev_sIE, double* dev_sI, double* dev_sEI, double* dev_GEE, double* dev_GIE, double* dev_GII, double* dev_GEI, double* dev_tmp1, double* dev_tmp2, double* dev_tmp3, double* dev_tmp4, int idt, int seqN, double* dev_vE, double* dev_vI, double* dev_lfp, double* dev_vbarE, double* dev_vbarI, double* dev_vegE, double* dev_vegI, int Eeg, int Ieg, int opt_storecurrs, double* dev_isynbarEtoE, double* dev_isynbarItoE, int lfp_size, double pVrevE, double pVrevI, double pglE, double pElE, double pslpE, double pVtE, double pCE, double paE, double ptwE, double pVrE, double pbE, double pglI, double pElI, double pslpI, double pVtI, double pCI, double paI, double ptwI, double pVrI, double pbI, double* dev_wE, double* dev_wI, double* dev_Enoise, double* dev_Inoise, int dev_Enoise_size_1, int dev_Inoise_size_1, double dev_dt, double* dev_erE, double* dev_edE, double* dev_erEI, double* dev_edEI, double* dev_erI, double* dev_edI, double* dev_erIE, double* dev_edIE, double dev_fdE, double dev_frE, double dev_fdEI, double dev_frEI, double dev_pvsE, double dev_pvsEI, double dev_fdI, double dev_frI, double dev_fdIE, double dev_frIE, double dev_pvsI, double dev_pvsIE, double* dev_tspEtimes, double* dev_tspEcelln, double* dev_tspItimes, double* dev_tspIcelln, int* dev_tspE_count, int* dev_tspI_count, double* dev_t) {
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    double Isyn, Iapp, Iion, dvdt, dwdt, wEst, vEst;
    vEst = dev_vE[idx];
    wEst = dev_wE[idx];
    Isyn = (dev_tmp1[idx] * (dev_vE[idx] - pVrevE)) + (dev_tmp2[idx] * (dev_vE[idx] - pVrevI));
    Iapp = dev_Enoise[idx * dev_Enoise_size_1 + (idt - 1)];
    Iion = ((-1) * pglE * (dev_vE[idx] - pElE)) + (pglE * pslpE * exp((dev_vE[idx] - pVtE) / pslpE)) - (dev_wE[idx]);
    dvdt = ((Iapp + (Iion) - Isyn) / pCE);
    dwdt = ((paE * (dev_vE[idx] - pElE) - dev_wE[idx]) / ptwE);
    dev_vE[idx] = dev_vE[idx] + (dev_dt * dvdt);
    dev_wE[idx] = dev_wE[idx] + (dev_dt * dwdt);

    // syn gates evolution
    dev_edE[idx] *= dev_fdE;
    dev_erE[idx] *= dev_frE;
    dev_sE[idx] = dev_erE[idx] - dev_edE[idx];
    dev_edEI[idx] *= dev_fdEI;
    dev_erEI[idx] *= dev_frEI;
    dev_sEI[idx] = dev_erEI[idx] - dev_edEI[idx];


    if (vEst >= 0) {
        // update dynamic vars
        dev_vE[idx] = pVrE;
        dev_wE[idx] = wEst + pbE;
        // update syn gates
        dev_edE[idx] += 1 / dev_pvsE;
        dev_erE[idx] += 1 / dev_pvsE;
        dev_edEI[idx] += 1 / dev_pvsEI;
        dev_erEI[idx] += 1 / dev_pvsEI;
    }
}

__global__ void kernelI(int NE, int NI, double* dev_sE, double* dev_sIE, double* dev_sI, double* dev_sEI, double* dev_GEE, double* dev_GIE, double* dev_GII, double* dev_GEI, double* dev_tmp1, double* dev_tmp2, double* dev_tmp3, double* dev_tmp4, int idt, int seqN, double* dev_vE, double* dev_vI, double* dev_lfp, double* dev_vbarE, double* dev_vbarI, double* dev_vegE, double* dev_vegI, int Eeg, int Ieg, int opt_storecurrs, double* dev_isynbarEtoE, double* dev_isynbarItoE, int lfp_size, double pVrevE, double pVrevI, double pglE, double pElE, double pslpE, double pVtE, double pCE, double paE, double ptwE, double pVrE, double pbE, double pglI, double pElI, double pslpI, double pVtI, double pCI, double paI, double ptwI, double pVrI, double pbI, double* dev_wE, double* dev_wI, double* dev_Enoise, double* dev_Inoise, int dev_Enoise_size_1, int dev_Inoise_size_1, double dev_dt, double* dev_erE, double* dev_edE, double* dev_erEI, double* dev_edEI, double* dev_erI, double* dev_edI, double* dev_erIE, double* dev_edIE, double dev_fdE, double dev_frE, double dev_fdEI, double dev_frEI, double dev_pvsE, double dev_pvsEI, double dev_fdI, double dev_frI, double dev_fdIE, double dev_frIE, double dev_pvsI, double dev_pvsIE, double* dev_tspEtimes, double* dev_tspEcelln, double* dev_tspItimes, double* dev_tspIcelln, int* dev_tspE_count, int* dev_tspI_count, double* dev_t) {
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    double Isyn, Iapp, Iion, dvdt, dwdt, wIst, vIst;
    vIst = dev_vI[idx];
    wIst = dev_wI[idx];
    Isyn = (dev_tmp3[idx] * (dev_vI[idx] - pVrevI)) + (dev_tmp4[idx] * (dev_vI[idx] - pVrevE));
    Iapp = dev_Inoise[idx * dev_Enoise_size_1 + (idt - 1)]; // should be dev_Inoise_size_1?
    Iion = ((-1) * pglI * (dev_vI[idx] - pElI)) + (pglI * pslpI * exp((dev_vI[idx] - pVtI) / pslpI)) - (dev_wI[idx]);

    dvdt = ((Iapp + Iion - Isyn) / pCI);
    dwdt = ((paI * (dev_vI[idx] - pElI) - dev_wI[idx]) / ptwI);
    dev_vI[idx] += dev_dt * dvdt;
    dev_wI[idx] += dev_dt * dwdt;

    // syn gates evolution
    dev_edI[idx] *= dev_fdI;
    dev_erI[idx] *= dev_frI;
    dev_sI[idx] = dev_erI[idx] - dev_edI[idx];
    dev_edIE[idx] *= dev_fdIE;
    dev_erIE[idx] *= dev_frIE;
    dev_sIE[idx] = dev_erIE[idx] - dev_edIE[idx];


    if (vIst >= 0) {
        // update dynamic vars
        dev_vI[idx] = pVrI;
        dev_wI[idx] = wIst + pbI;
        // update syn gates
        dev_edI[idx] += 1 / dev_pvsI;
        dev_erI[idx] += 1 / dev_pvsI;
        dev_edIE[idx] += 1 / dev_pvsIE;
        dev_erIE[idx] += 1 / dev_pvsIE;
    }
}

void NetworkRunSeqt(struct pm p, struct inpseq inps, int NE, int NI, double T, struct options opt) {

    /*outputs*/
    struct Conn conn;
    struct Vbar vbar;
    struct Veg veg;
    struct vec lfp;
    struct Tsp tspE;
    struct Tsp tspI;
    struct Isynbar isynbar;
    struct Inp inp;
    double* seqs;

    int max_size_tsp = 15000;
    tspE.times.size = max_size_tsp; // max size
    tspE.times.val = (double *)malloc(tspE.times.size * sizeof(double));
    tspE.celln.size = max_size_tsp; // max size
    tspE.celln.val = (double*)malloc(tspE.celln.size * sizeof(double));
    int tspE_count = 0;

    tspI.times.size = max_size_tsp; // max size
    tspI.times.val = (double*)malloc(tspI.times.size * sizeof(double));
    tspI.celln.size = max_size_tsp; // max size
    tspI.celln.val = (double*)malloc(tspI.celln.size * sizeof(double));
    int tspI_count = 0;

    /* pre select the sequence */
    //int NEseq[] = { 66, 153, 171, 314, 371, 408, 503, 604, 717, 798 };
    int* NEseq;
    int nL = 10;
    NEseq = (int*)malloc(nL * sizeof(int));
    seqs = (double*)malloc(nL * sizeof(double));
    for (int i = 0; i < nL; i++) {
        NEseq[i] = ((rand() % 80) + 1) + (i * 80);
        seqs[i] = NEseq[i];
    }

    qsort(seqs, nL, sizeof(double), AscendCmp);



    /* opt options: nonoise novar noiseprc */
    p.gnoiseE = opt.nonoise ? 0 : p.gnoiseE * (opt.noiseprc / 100);
    p.gnoiseI = opt.nonoise ? 0 : p.gnoiseI * (opt.noiseprc / 100);

    double* Edc_dist;
    Edc_dist = (double*)malloc(NE * sizeof(double));
    //double Edc_dist[] = { 34.6005, 52.1397, 42.9016, 39.7478, 42.859, 39.1801, 39.5034, 45.9588, 45.6361, 45.6688, 42.686, 35.1701, 42.869, 46.5209, 41.9556, 44.1388, 42.9075, 38.7862, 41.1755, 36.8509, 43.5536, 35.4117, 35.7245, 36.762, 28.2229, 45.7535, 41.3008, 36.9803, 45.4812, 33.1539, 39.591, 39.0342, 41.2768, 41.2514, 36.5405, 39.8798, 39.3405, 42.5108, 44.3731, 44.4371, 36.5454, 40.3094, 35.1435, 35.546, 39.9726, 46.1305, 36.9213, 41.4855, 39.0977, 44.4694, 35.6437, 40.1302, 42.2101, 44.4024, 46.1768, 40.3437, 34.0336, 37.0308, 35.7537, 49.4018, 37.5376, 42.9923, 39.2303, 43.5544, 36.9406, 34.3909, 34.3105, 41.9528, 39.2905, 39.2158, 45.6772, 41.1663, 40.7912, 46.3508, 36.7821, 42.7865, 43.3404, 39.0251, 40.8627, 35.3366, 35.4082, 40.4195, 42.889, 50.342, 37.3324, 40.7493, 39.67, 32.2679, 38.2441, 32.8213, 43.3615, 36.4479, 40.4004, 37.8219, 41.2141, 37.5987, 41.9599, 42.9575, 46.8476, 39.2235, 31.4466, 36.6416, 45.4184, 35.7114, 43.8438, 40.4962, 45.7468, 32.1564, 39.2092, 35.1686, 51.632, 43.3009, 45.5159, 35.7673, 38.1255, 38.9101, 44.3937, 38.8885, 42.8062, 31.7927, 38.5846, 36.7057, 33.6918, 42.0319, 41.1279, 40.1339, 34.6653, 44.51, 41.4007, 38.8037, 40.0916, 38.952, 32.9992, 38.8574, 36.6745, 36.0832, 35.3744, 37.8658, 31.9895, 43.8569, 42.0802, 39.9199, 39.8609, 36.8073, 44.0747, 39.4671, 37.1419, 45.4055, 39.1009, 37.6439, 38.825, 36.6083, 35.5195, 50.104, 46.622, 41.2301, 34.9715, 36.5381, 39.2939, 43.1657, 34.672, 30.6805, 34.2036, 41.334, 41.5654, 41.8067, 39.4789, 40.7348, 38.0954, 43.4481, 34.5532, 41.8201, 36.6052, 38.6605, 42.2111, 44.1564, 35.5294, 45.0426, 42.6406, 39.7285, 39.2191, 39.1296, 38.7876, 40.0922, 40.2052, 43.3043, 46.1079, 41.8677, 39.1611, 42.5008, 40.7329, 35.8809, 43.7969, 41.2282, 40.5407, 42.061, 41.0456, 36.2341, 39.3506, 39.4158, 37.872, 46.7284, 36.4971, 38.0647, 37.152, 35.3032, 39.231, 38.9037, 46.1203, 39.0039, 35.7431, 46.4138, 44.9387, 39.0815, 33.9754, 38.2215, 39.3762, 41.1043, 38.9553, 41.7737, 41.5676, 34.9973, 36.2082, 37.0356, 37.9687, 38.7177, 40.0499, 27.8833, 38.1719, 44.9698, 35.7332, 43.7349, 41.4013, 39.884, 40.7298, 33.7398, 39.6618, 46.4158, 40.3934, 40.1655, 37.0633, 39.8767, 40.9294, 41.7056, 38.5088, 39.0542, 48.0948, 30.9666, 48.9178, 41.3503, 44.0002, 33.3433, 37.6399, 38.8877, 41.6909, 33.3192, 41.8865, 35.1486, 40.2648, 42.6094, 41.3082, 44.3305, 44.0243, 37.3964, 41.0282, 36.2225, 34.7128, 43.6993, 40.0002, 39.7803, 43.6445, 42.3783, 41.4008, 45.001, 43.7192, 40.9591, 37.2386, 37.3938, 44.7684, 33.5527, 39.9022, 32.2046, 44.082, 43.4469, 40.0046, 39.7167, 30.0549, 42.3247, 31.2303, 30.7229, 40.3197, 36.2061, 41.646, 42.7079, 43.4309, 37.2354, 41.7975, 40.4025, 43.3043, 42.1446, 43.5916, 39.4722, 39.4112, 44.0311, 31.5054, 37.9817, 34.9176, 38.4697, 42.5947, 43.3029, 35.9402, 38.1157, 40.5481, 38.8325, 41.2073, 41.5997, 36.2802, 39.2927, 31.4716, 44.5814, 37.4836, 35.1846, 38.9842, 34.2854, 39.9166, 37.7573, 48.7111, 44.5539, 30.0125, 41.7653, 34.4074, 38.9798, 40.6576, 42.9909, 38.9078, 46.3052, 38.0763, 41.31, 42.6589, 40.3408, 43.5238, 41.2929, 36.8634, 32.7785, 47.4344, 37.5819, 40.4134, 42.2527, 40.4544, 36.3811, 38.1291, 39.5004, 45.9158, 36.5567, 43.1387, 41.2345, 39.0646, 35.7721, 38.8634, 39.6532, 34.1224, 40.7687, 36.7108, 39.623, 41.3449, 36.3814, 38.847, 41.4003, 32.6566, 44.1439, 49.6978, 43.8376, 38.7369, 41.7145, 35.8561, 47.5115, 43.7628, 43.1494, 36.4965, 41.2798, 37.7668, 38.7543, 37.72, 35.8971, 36.365, 39.1604, 33.2045, 42.4304, 39.5288, 42.7966, 41.0786, 41.9771, 34.0675, 35.9189, 38.212, 40.4386, 44.5149, 38.8401, 45.0462, 41.9017, 44.6965, 40.5078, 37.3727, 34.0744, 40.622, 43.2742, 38.8296, 37.8369, 38.7654, 35.6136, 38.028, 39.277, 40.1834, 39.7449, 42.4453, 40.4373, 47.2561, 41.2481, 47.218, 37.1075, 42.1062, 38.959, 42.4006, 42.3757, 31.2559, 34.6918, 34.2359, 41.6074, 45.8808, 38.6927, 43.2493, 42.1822, 35.7935, 41.5899, 36.9924, 46.0651, 39.8697, 46.544, 38.2998, 42.3577, 39.7488, 31.9122, 36.0715, 42.45, 39.7805, 35.5251, 37.4945, 40.9981, 36.0279, 43.8998, 37.4372, 47.2355, 35.6805, 40.7968, 33.9159, 37.1055, 37.627, 41.6053, 43.7685, 41.2019, 38.5077, 43.262, 43.1955, 40.4808, 42.285, 41.6512, 36.0522, 43.0383, 37.3712, 37.5843, 40.7078, 38.77, 39.4727, 42.3814, 44.1873, 39.2082, 41.3107, 39.0468, 40.9184, 41.76, 37.5325, 41.0993, 42.4044, 40.3692, 46.9194, 37.5658, 37.0518, 33.0005, 43.6419, 43.4683, 39.6804, 43.5939, 40.7348, 41.1632, 40.4518, 41.7598, 40.4066, 51.1493, 35.3333, 32.5828, 35.4373, 35.6266, 38.2656, 39.3261, 39.1259, 42.1653, 41.5571, 43.0049, 47.113, 44.8923, 34.867, 30.6842, 43.6077, 32.6574, 40.267, 40.1419, 48.9087, 39.7231, 37.9707, 40.9432, 40.9832, 40.2802, 37.5657, 35.1096, 41.266, 34.6285, 35.8713, 45.3249, 38.3244, 39.4387, 43.5993, 38.7996, 44.1175, 38.6197, 44.0512, 42.5173, 39.1479, 36.5372, 35.8276, 38.9197, 38.2474, 38.3653, 43.9342, 38.8092, 44.5747, 37.8735, 43.8903, 37.911, 40.7063, 43.883, 38.3441, 38.2469, 48.0136, 43.804, 38.272, 42.5958, 38.5597, 42.8235, 45.6634, 33.5819, 44.1154, 45.8319, 40.1899, 46.985, 40.6216, 35.0515, 31.226, 38.6664, 42.8542, 41.2696, 41.6544, 37.6917, 40.576, 33.4453, 36.9596, 36.7248, 42.0789, 39.9434, 35.3779, 39.9619, 37.2408, 37.3332, 43.4566, 40.4537, 41.5935, 43.5359, 40.721, 42.2034, 42.7319, 44.6824, 41.9034, 45.6489, 40.0904, 39.8085, 46.8053, 37.9612, 39.9886, 43.6795, 40.5992, 45.6197, 44.1365, 41.1663, 36.8892, 42.2668, 34.4695, 40.9779, 43.2338, 40.8522, 43.5187, 48.1555, 43.6957, 41.0677, 42.5666, 41.7019, 34.7411, 38.3344, 44.8988, 39.8257, 42.3297, 35.974, 40.2581, 42.4012, 34.5539, 41.3904, 39.2726, 36.2419, 39.8499, 32.4148, 31.4881, 35.2923, 36.0379, 35.3079, 33.0983, 41.1529, 33.6233, 40.4409, 43.1483, 39.9911, 40.3724, 38.4874, 34.0693, 39.8247, 43.8433, 46.953, 38.2792, 33.4907, 40.6654, 41.5051, 39.0922, 35.4044, 48.0973, 30.5619, 37.9601, 34.7135, 37.4555, 41.2714, 40.5522, 37.1571, 43.108, 42.4896, 42.5895, 38.2975, 44.1943, 42.6428, 50.0351, 44.2538, 44.6277, 40.2119, 34.8465, 38.5151, 36.9688, 37.7441, 42.2206, 37.7729, 36.4195, 38.3627, 39.3565, 41.6373, 36.1895, 41.2693, 40.3121, 45.2975, 39.1473, 39.4621, 35.3146, 34.4589, 41.242, 39.002, 42.015, 36.4294, 47.634, 40.4889, 44.1881, 39.0923, 39.35, 42.7602, 42.223, 35.519, 33.8692, 35.6085, 34.3369, 40.2383, 38.355, 38.528, 34.5561, 43.1183, 41.7576, 39.6415, 44.0847, 36.5041, 41.6588, 41.3938, 41.397, 37.083, 41.3074, 37.9405, 36.4142, 35.1869, 44.1513, 36.6162, 39.3083, 35.1654, 38.8115, 27.0718, 35.6522, 34.2943, 35.9422, 39.1469, 38.6986, 47.7776, 37.7129, 38.9999, 33.7227, 38.0905, 34.6481, 40.1212, 43.4123, 41.617, 37.1975, 33.4778, 45.8401, 48.2002, 40.482, 36.0404, 44.7911, 37.6294, 38.1208, 43.5455, 34.4591, 32.173, 41.6827, 41.603, 40.3806, 41.9867, 44.329, 43.8818, 37.7257, 43.2399, 40.693, 37.9778, 35.2268, 42.5879, 38.5855, 40.1857, 36.8282, 33.7979, 40.6863, 39.7514, 44.7961, 43.2068, 44.2132, 37.0045, 36.2547, 34.9237, 41.9919, 51.1563, 42.9103, 36.9077, 43.3465, 35.4867, 34.3021, 42.8698, 36.8884, 41.2639, 45.6261, 41.6045, 43.7186, 33.5768, 42.6461, 48.554, 42.1646, 33.8365, 39.1874, 38.0001, 41.5321, 41.6481, 41.622, 38.5449, 37.6029, 37.6416 };
    double Edc_dist_max = log(0); //= -inf
    for (int i = 0; i < NE; i++) {
        Edc_dist[i] = (RandNormal() * p.DCstdE * p.Edc) + p.Edc;
        Edc_dist_max = Edc_dist[i] > Edc_dist_max ? Edc_dist[i] : Edc_dist_max;
    }
    double* Idc_dist;
    Idc_dist = (double*)malloc(NI * sizeof(double));
    //double Idc_dist[] = { 195.3637, 146.6459, 176.2685, 184.8668, 168.2501, 188.5901, 178.7162, 163.1106, 182.9045, 175.1727, 172.6223, 167.1962, 181.106, 146.7697, 172.83, 170.2161, 163.5858, 191.7486, 166.7831, 189.7314, 197.5651, 177.1763, 185.0004, 191.5113, 178.5424, 189.7357, 157.2738, 199.9876, 162.1879, 147.081, 204.921, 178.8709, 188.0806, 173.4613, 161.6295, 124.6862, 191.273, 174.8397, 176.4478, 187.3009, 154.4517, 166.87, 200.6519, 190.7616, 156.9369, 140.3412, 169.7176, 183.8519, 196.9628, 181.6871, 159.7984, 185.5108, 158.898, 162.7026, 168.2328, 157.8709, 175.1226, 163.8009, 174.8576, 171.6764, 172.6239, 170.9363, 202.1993, 190.9855, 181.0633, 153.595, 150.7355, 144.6345, 226.8935, 197.5027, 184.6257, 162.4637, 159.3654, 189.8575, 208.1715, 149.5198, 171.9108, 178.4827, 144.1441, 195.1424, 172.5361, 214.4193, 172.9638, 187.3653, 159.4363, 168.7525, 158.963, 187.0664, 203.4331, 169.3144, 187.8548, 170.9215, 181.8379, 201.5325, 182.1651, 161.3368, 164.5721, 176.9423, 176.55, 164.4153, 183.252, 202.7975, 175.479, 176.3177, 140.3726, 166.0588, 154.9211, 173.0478, 189.4606, 207.4188, 212.3729, 177.8961, 174.2365, 194.7153, 188.8229, 193.7745, 194.009, 153.3545, 189.7266, 178.3523, 166.3155, 167.5153, 203.0662, 165.4247, 157.7373, 183.8644, 216.1939, 180.46, 185.5494, 163.1116, 210.1359, 182.2498, 189.5418, 162.8628, 195.3728, 187.0046, 159.192, 180.7153, 171.8892, 181.9665, 175.49, 176.5818, 161.4076, 174.1807, 193.7975, 211.4041, 159.1106, 222.7934, 207.4694, 183.0331, 174.5783, 167.4242, 194.9899, 167.4971, 171.6861, 195.9051, 187.847, 196.1415, 189.0852, 172.7839 };
    for (int i = 0; i < NI; i++) {
        Idc_dist[i] = (RandNormal() * p.DCstdI * p.Idc) + p.Idc;
    }

    double* w;
    w = (double*)malloc(nL * sizeof(double));
    //double w[] = { -0.51385, 0.79637, -0.67119, 1.1867, 0.7907, 0.28772, 0.0032261, 0.36562, 3.5267, -0.11244 };
    for (int i = 0; i < nL; i++) {
        w[i] = RandNormal();
    }
    qsort(w, nL, sizeof(double), DescendCmp);
    for (int i = 0; i < nL; i++) {
        Edc_dist[NEseq[i]] = (Edc_dist_max + (p.dcbias * p.DCstdE * p.Edc)) * ((double)1 + (w[i] * (double)0.02));
    }
    inp.Edc = Edc_dist; //on matlab is transposed
    inp.Idc = Idc_dist;

    free(w);


    /* inputs */
    struct matrix MX; //matrix to sum only some incoming inputs of CA3
    MX.size[0] = 100;
    MX.size[1] = NE;
    MX.val = (double*)malloc(MX.size[0] * MX.size[1] * sizeof(double));
    for (int i = 0; i < MX.size[0]; i++) {
        for (int j = 0; j < MX.size[1]; j++) {
            MX.val[i * MX.size[1] + j] = 0;
        }
    }
    int kkS;
    for (int i = 0; i < nL; i++) {
        kkS = ceil(((float)(NEseq[i] - 1) / NE) * 100) - 1;
        MX.val[kkS * MX.size[1] + (NEseq[i] - 1)] = 1;
    }
    transpose(&MX);  // after it will be used transposed
    free(NEseq);

    /* synapses */
    printf("wire ntwk - ");
    clock_t tic = clock();
    if (opt.novar) {
        p.gvarEE = 0;
        p.gvarII = 0;
        p.gvarEI = 0;
        p.gvarIE = 0;
    }

    double mn = p.gmaxEE / NE;
    double vr = p.gvarEE * mn;
    struct matrix GEE;
    GEE.size[0] = NE;
    GEE.size[1] = NE;
    GEE.val = (double*)malloc(GEE.size[0] * GEE.size[1] * sizeof(double));
    double g;
    for (int i = 0; i < GEE.size[0]; i++) {
        for (int j = 0; j < GEE.size[1]; j++) {
            g = RandNormal() * sqrt(vr) + mn;
            GEE.val[i * GEE.size[1] + j] = g > 0 ? g : 0;
        }
    }
    //FILE* rfp = NULL;
    //rfp = fopen("C:\\Users\\mclab\\Desktop\\simone\\thesis\\ripples-parallel\\ripples-parallel\\bin\\GEE.bin", "rb");
    //fread(GEE.val, sizeof(double), GEE.size[0] * GEE.size[1], rfp);
    //fclose(rfp);

    conn.EtoE = GEE;
    transpose(&GEE); // after is used only transposed

    mn = p.gmaxII / NI;
    vr = p.gvarII * mn;
    struct matrix GII;
    GII.size[0] = NI;
    GII.size[1] = NI;
    GII.val = (double*)malloc(GII.size[0] * GII.size[1] * sizeof(double));
    for (int i = 0; i < GII.size[0]; i++) {
        for (int j = 0; j < GII.size[1]; j++) {
            g = RandNormal() * sqrt(vr) + mn;
            GII.val[i * GII.size[1] + j] = g > 0 ? g : 0;
        }
    }
    //rfp = NULL;
    //rfp = fopen("C:\\Users\\mclab\\Desktop\\simone\\thesis\\ripples-parallel\\ripples-parallel\\bin\\GII.bin", "rb");
    //fread(GII.val, sizeof(double), GII.size[0] * GII.size[1], rfp);
    //fclose(rfp);

    conn.ItoI = GII;


    mn = p.gmaxEI / NE;
    vr = p.gvarEI * mn;
    struct matrix GEI;
    GEI.size[0] = NE;
    GEI.size[1] = NI;
    GEI.val = (double*)malloc(GEI.size[0] * GEI.size[1] * sizeof(double));
    for (int i = 0; i < GEI.size[0]; i++) {
        for (int j = 0; j < GEI.size[1]; j++) {
            g = RandNormal() * sqrt(vr) + mn;
            GEI.val[i * GEI.size[1] + j] = g > 0 ? g : 0;
        }
    }
    //rfp = NULL;
    //rfp = fopen("C:\\Users\\mclab\\Desktop\\simone\\thesis\\ripples-parallel\\ripples-parallel\\bin\\GEI.bin", "rb");
    //fread(GEI.val, sizeof(double), GEI.size[0] * GEI.size[1], rfp);
    //fclose(rfp);

    conn.EtoI = GEI;

    mn = p.gmaxIE / NI;
    vr = p.gvarIE * mn;
    struct matrix GIE;
    GIE.size[0] = NI;
    GIE.size[1] = NE;
    GIE.val = (double*)malloc(GIE.size[0] * GIE.size[1] * sizeof(double));
    for (int i = 0; i < GIE.size[0]; i++) {
        for (int j = 0; j < GIE.size[1]; j++) {
            g = RandNormal() * sqrt(vr) + mn;
            GIE.val[i * GIE.size[1] + j] = g > 0 ? g : 0;
        }
    }
    //rfp = NULL;
    //rfp = fopen("C:\\Users\\mclab\\Desktop\\simone\\thesis\\ripples-parallel\\ripples-parallel\\bin\\GIE.bin", "rb");
    //fread(GIE.val, sizeof(double), GIE.size[0] * GIE.size[1], rfp);
    //fclose(rfp);

    conn.ItoE = GIE;
    transpose(&GIE); // after is used only transposed

    clock_t toc = clock();
    printf("elapsed time is %.2lf seconds.\n", (double)(toc - tic) / CLOCKS_PER_SEC);


    /* initialize sim */


    // time
    double dt = 0.001; // [=]ms integration step

    // allocate simulation ouput(GIE' * sIE)' * (vE - VrevI)
    int s = (ceil(1000 / dt) - 1) / 1000 + 1 + 1000 * (T - 1) + 1;
    vbar.E = (double*)malloc(s * sizeof(double));
    vbar.I = (double*)malloc(s * sizeof(double));
    veg.E = (double*)malloc(s * sizeof(double));
    veg.I = (double*)malloc(s * sizeof(double));
    lfp.size = s;
    lfp.val = (double*)malloc(s * sizeof(double));
    for (int i = 0; i < s; i++) {
        vbar.E[i] = 0;
        vbar.I[i] = 0;
        veg.E[i] = 0;
        veg.I[i] = 0;
        lfp.val[i] = 0;
    }

    // t = 0:dt:1000
    struct vec t; // one sec time axis 
    s = ceil(1000 / dt);
    t.size = s;
    t.val = (double*)malloc(s * sizeof(double));
    for (int i = 0; dt * i <= 1000; i++) {
        t.val[i] = dt * i;
    }

    // peak values of biexps signals
    double pvsE = exp((1 / (1 / p.tauEd - 1 / p.tauEr) * log(p.tauEd / p.tauEr)) / p.tauEr) - exp((1 / (1 / p.tauEd - 1 / p.tauEr) * log(p.tauEd / p.tauEr)) / p.tauEd);
    double fdE = exp(-dt / p.tauEd); //factor of decay
    double frE = exp(-dt / p.tauEr); //factor of rise

    double pvsI = exp((1 / (1 / p.tauId - 1 / p.tauIr) * log(p.tauId / p.tauIr)) / p.tauIr) - exp((1 / (1 / p.tauId - 1 / p.tauIr) * log(p.tauId / p.tauIr)) / p.tauId);
    double fdI = exp(-dt / p.tauId);
    double frI = exp(-dt / p.tauIr);

    double pvsIE = exp((1 / (1 / p.tauIEd - 1 / p.tauIEr) * log(p.tauIEd / p.tauIEr)) / p.tauIEr) - exp((1 / (1 / p.tauIEd - 1 / p.tauIEr) * log(p.tauIEd / p.tauIEr)) / p.tauIEd);
    double fdIE = exp(-dt / p.tauIEd);
    double frIE = exp(-dt / p.tauIEr);

    double pvsEI = exp((1 / (1 / p.tauEId - 1 / p.tauEIr) * log(p.tauEId / p.tauEIr)) / p.tauEIr) - exp((1 / (1 / p.tauEId - 1 / p.tauEIr) * log(p.tauEId / p.tauEIr)) / p.tauEId);
    double fdEI = exp(-dt / p.tauEId);
    double frEI = exp(-dt / p.tauEIr);

    int Eeg = (rand() % NE);
    //int Eeg = 12;
    veg.ne = Eeg;
    int Ieg = (rand() % NI);
    //int Ieg = 59;
    veg.ni = Ieg;

    s = (ceil(1000 / dt) - 1) / 1000 + 1 + 1000 * (T - 1) + 1;
    isynbar.ItoE.size[0] = NE;
    isynbar.ItoE.size[1] = s;
    isynbar.ItoE.val = (double*)malloc(isynbar.ItoE.size[0] * isynbar.ItoE.size[1] * sizeof(double));
    isynbar.EtoE.size[0] = NE;
    isynbar.EtoE.size[1] = s;
    isynbar.EtoE.val = (double*)malloc(isynbar.EtoE.size[0] * isynbar.EtoE.size[1] * sizeof(double));
    if (opt.storecurrs) {
        for (int i = 0; i < isynbar.ItoE.size[0]; i++) {
            for (int j = 0; j < isynbar.ItoE.size[1]; j++) {
                isynbar.ItoE.val[i * isynbar.ItoE.size[1] + j] = 0;
            }
        }
        for (int i = 0; i < isynbar.EtoE.size[0]; i++) {
            for (int j = 0; j < isynbar.EtoE.size[1]; j++) {
                isynbar.EtoE.val[i * isynbar.EtoE.size[1] + j] = 0;
            }
        }
    }

    s = ceil(1 / dt) + 1;
    struct matrix Einptrace;
    Einptrace.size[0] = NE;
    Einptrace.size[1] = s;
    Einptrace.val = (double*)malloc(Einptrace.size[0] * Einptrace.size[1] * sizeof(double));
    for (int i = 0; i < Einptrace.size[0]; i++) {
        for (int j = 0; j < Einptrace.size[1]; j++) {
            Einptrace.val[i * Einptrace.size[1] + j] = 0;
        }
    }

    s = ceil(1 / dt) + 1;
    struct matrix Iinptrace;
    Iinptrace.size[0] = NI;
    Iinptrace.size[1] = s;
    Iinptrace.val = (double*)malloc(Iinptrace.size[0] * Iinptrace.size[1] * sizeof(double));
    for (int i = 0; i < Iinptrace.size[0]; i++) {
        for (int j = 0; j < Iinptrace.size[1]; j++) {
            Iinptrace.val[i * Iinptrace.size[1] + j] = 0;
        }
    }

    /* for each second of simulation we generate new noise */

    /* the noise is taken by a binary file */

    int seqN = 1;
    struct matrix Enoise;
    Enoise.size[0] = NE;
    Enoise.size[1] = (int)ceil(1000 / dt) + 1;
    struct matrix Inoise;
    Inoise.size[0] = NI;
    Inoise.size[1] = (int)ceil(1000 / dt) + 1;
    Enoise.val = (double*)malloc(Enoise.size[0] * Enoise.size[1] * sizeof(double));
    Inoise.val = (double*)malloc(Inoise.size[0] * Inoise.size[1] * sizeof(double));

    // for the initialization of the first instant of simulation
    struct vec vE, wE, sE, sEI, erE, edE, erEI, edEI;
    vE.size = NE;
    wE.size = NE;
    sE.size = NE;
    sEI.size = NE;
    erE.size = NE;
    edE.size = NE;
    erEI.size = NE;
    edEI.size = NE;

    vE.val = (double *)malloc(NE * sizeof(double));
    //double rtmp1[] = { -60.98514944929608, -50.57603118190275, -56.57040311395642, -60.58004542764260, -46.22603497608614, -66.34703389927067, -48.90667352764604, -66.25790974333846, -64.37054726650818, -52.20532043296545, -64.85673006305063, -53.91209292439533, -49.26126543570118, -46.83439804845644, -59.45981846481071, -59.91210412250814, -69.12154074371259, -46.09658681379807, -60.88022056869478, -66.52857654825858, -48.00076745436184, -58.15439476176215, -47.98826564838527, -61.50620935734737, -57.72168760698175, -53.36471791685754, -66.40061942131750, -62.66668537084679, -59.37391380060610, -55.04592133118825, -65.16650580307930, -57.74441661769551, -50.33686340254366, -46.27088610274045, -62.07465925272403, -51.43592016946068, -61.01855157430303, -48.25640021549141, -47.42128168761098, -55.83976639824181, -52.91275456139674, -61.58581484243555, -61.65221911726302, -60.25565196617281, -64.92532816337014, -58.44034318658018, -67.56341970691172, -53.21988312903471, -68.65076262647071, -52.52399436677342, -56.23719731766136, -64.74905037183444, -61.05413116198932, -47.26932316069399, -62.13546008185488, -58.37550063773523, -47.15973967289483, -47.87277902491827, -64.38738195607128, -46.99223739742038, -64.37797633010996, -69.84006325160433, -68.46988239186533, -63.15080668942876, -57.28010278663830, -47.73309001465172, -57.84085929367406, -56.87174292070030, -63.13352962010563, -46.95490153837527, -57.52370278508901, -50.58431658129447, -51.49398171420643, -52.15316466558382, -51.85457914899172, -53.68426550014953, -58.73919597291081, -48.32670532732640, -52.32043632140537, -47.99682373229948, -50.72752386672016, -55.15351401649481, -46.33393774372340, -61.22052757218631, -49.93978456799569, -48.04530471533180, -69.85756813786000, -46.28076094803289, -57.88286632763742, -57.79841347238015, -62.62750064628872, -61.12521485378353, -57.75490774860162, -53.89826507363228, -66.26478379556428, -58.45434059618984, -51.38980747786069, -61.26674484850781, -48.50566499445097, -64.38592501253315, -60.21162117778805, -59.61399373669448, -51.81716118422801, -67.22569417989237, -61.22979917445073, -46.26305044878875, -55.02727675912322, -69.89636277493473, -61.96252184190074, -64.66316646595506, -52.40737981640129, -64.52807848466243, -47.74505080385340, -50.82759775382799, -61.45939703758316, -56.09133485421339, -63.48455659301306, -56.69003587554510, -58.97004351372132, -69.94166339347071, -66.13612579891817, -58.16581880191068, -58.17721742819124, -69.46651483656301, -63.90882657967072, -59.59544090740625, -61.39783138018103, -68.80318311457000, -69.48529047503061, -55.24841933552682, -48.90091521476457, -58.05682021788840, -58.69606625738951, -46.69858011739176, -62.67834708943514, -59.83669425705428, -55.11263442226709, -57.04146019552888, -60.90635421642946, -59.01154709919456, -55.69741660765172, -61.70352941509935, -58.25628605829129, -47.61985462654545, -62.05315235537342, -52.10740250867497, -50.14444487826471, -52.46556343291405, -57.03359997284360, -55.90769510919670, -58.32504044701912, -60.49860534764615, -64.91012190861150, -59.81930716844096, -68.04903138390281, -46.16178648811473, -49.54547429683939, -65.30767676884321, -47.46059098458035, -69.38000166199397, -62.09059252363458, -62.25867727534003, -68.15316575663466, -51.74128505950478, -62.81230358079635, -48.39025105570775, -59.46708317798657, -59.56113785972994, -63.96585102885069, -48.71880030668266, -56.78265747237034, -60.76937689465299, -64.15626437682863, -69.77337809021991, -55.06517855005353, -65.04913258504516, -48.19591151026788, -60.24647301008873, -49.58251613271560, -58.91687353680398, -68.30248627746204, -69.72119703165119, -59.99270898587378, -64.05904293275485, -58.07354105686378, -48.78241821885094, -54.08834778290813, -51.89093202000871, -57.06727701178932, -49.42717849194131, -46.63080127448953, -60.56382589978647, -48.72083356384321, -62.53702320334723, -65.17857597208193, -63.57702649413992, -55.44315919770031, -47.10371003443724, -52.20655164871290, -65.50691152584709, -68.39584326631039, -69.93362421101197, -62.82447799796341, -61.70116780097048, -49.24846924881118, -50.40328631098729, -54.32249110095127, -61.22216880559171, -65.39415249889538, -61.10123932245085, -61.69321963954020, -50.40644286213298, -48.29622653594303, -49.96897566081778, -57.40691956557856, -60.25994996386916, -63.70957790354909, -64.08807964138808, -50.66426834881780, -53.04951129462269, -49.99636514414902, -52.38315188680095, -48.97107650041664, -64.72627286581367, -64.96786675940535, -46.36223523523820, -46.11853174700968, -52.68485444005697, -46.15948534265020, -69.21165740292653, -59.35579306843047, -54.36208598059986, -49.78482279207631, -63.68109135783349, -66.22198127305855, -49.89459325380514, -60.82605558820046, -50.86755269486517, -51.67260542959602, -61.61237261570705, -64.67307769135056, -66.48162425745083, -68.13103392046841, -62.18398610540638, -58.78696570601610, -50.81240289415324, -54.48847859768149, -59.39810748218662, -54.25463956817877, -63.00323655329634, -52.39763884687497, -60.01810206927552, -48.24005337489317, -52.62051646689156, -60.94835566908554, -49.43437111163674, -67.59679749373052, -52.58624304668884, -69.45547606199537, -66.42660741346918, -56.16999473938518, -47.70331270734176, -51.59098681380803, -69.04412739836272, -58.48096570942003, -49.28732739234682, -59.86631757156484, -67.60010732945445, -66.38313735523583, -55.32597117688540, -68.67591209550676, -58.56783697259250, -58.05610196815799, -55.67868861159157, -58.95617284880898, -62.72598065632583, -56.20223016633536, -55.76025097080457, -60.36492040457168, -52.17009458744627, -67.52291053458190, -53.55313361401031, -58.56912600747755, -50.00181991746257, -60.04083072442801, -63.28156406703408, -48.31948975504493, -46.68635623347804, -67.97820793654348, -54.88494190294783, -46.52596665554628, -67.80782915349140, -53.24510123709481, -50.66501395037322, -52.98814935942438, -49.05088160628981, -62.66546516024559, -46.77072891482138, -64.02880345033196, -65.50190186118064, -52.53165889924276, -61.35297041703639, -54.16304138437042, -62.39554749027435, -56.26299462857126, -69.87585098094247, -48.99881708339124, -53.90431469508485, -62.72781018598678, -59.39159806433086, -55.98828990839031, -55.16259333748597, -55.57748475575417, -55.12144422911955, -54.46426547819375, -66.57401219400239, -59.54810222269551, -57.39625104206143, -63.56684266327461, -51.13351394454775, -47.97382781529161, -57.66864190639071, -61.23670108712552, -65.52849978244247, -51.77805975890433, -58.16228711901347, -49.85964548891662, -52.33252399901372, -47.25273010517034, -69.64547926261898, -58.77384867342592, -48.61213216932875, -68.17126536840456, -60.48442275842613, -48.55659610553623, -52.14740320540860, -63.22202072282022, -58.65285184054191, -66.00115395260521, -60.13574332257490, -50.12873058147245, -52.18774401866008, -52.07774539162580, -66.37977592764955, -51.33680119066759, -55.57098131073920, -52.03179253351375, -53.54725047073065, -62.49523313471139, -59.58558211422011, -68.50438460642579, -64.25587956686860, -66.13608800615229, -46.91692595512046, -47.79935702929871, -55.30497500923681, -67.75220841766630, -51.02436996305141, -63.09146701016719, -50.10806524881330, -61.19996498507462, -60.60048347224507, -46.25303885352344, -59.34281817429774, -59.19359675466838, -61.24503183171109, -56.68086784086938, -57.32419819671792, -53.04309129382958, -64.60389937007798, -61.24004268051330, -68.73743738231398, -65.08880343262464, -52.66253235254720, -46.29618819817063, -58.03762150310006, -58.84522958105194, -51.30566243713783, -63.61976914884804, -49.51345624317383, -60.84611717771243, -64.25912969354304, -53.20401866941664, -57.35697941533381, -57.73361397635857, -53.90125886008000, -66.47690137886576, -56.21384846710256, -53.59422033667776, -55.34619072984880, -69.84093714767677, -47.32139657306195, -50.88282280633142, -64.03518262001731, -66.57908724771916, -47.86606629006567, -63.74018296400053, -56.13275790532347, -56.82160950390150, -57.94065965453073, -55.22679152231275, -47.59127512347906, -67.97642784393749, -48.43722191010635, -64.11596021803257, -49.49830578424550, -54.40172369193218, -47.88247736691876, -52.18797441575740, -60.58286294341108, -54.45415458516352, -62.12980501797851, -63.80415973417269, -53.95855969158701, -69.84875586569581, -68.02268676079554, -59.64868851535823, -65.42871558997440, -49.41387971396428, -58.64697767073247, -64.00987665107172, -55.76855147385548, -47.28776672219450, -69.09156461950285, -69.38094828662599, -66.00041887079622, -56.17678487525592, -54.21912341848333, -58.44619579879631, -48.12302922995233, -64.40207201357384, -48.16042740953308, -55.16216724789931, -59.50215620615835, -46.13370295741232, -67.17182201312110, -55.73818676857793, -68.63893062129870, -48.22093299572367, -54.82653901022307, -51.95507487852061, -67.87496044090788, -65.28505193045103, -59.58491817163684, -64.14912561886136, -64.96091668124805, -59.11944262680566, -61.15740963552614, -49.27155967271929, -65.56773404300978, -60.52260437918489, -63.25673357483615, -56.02067742958632, -59.42011348796211, -62.44729169737113, -47.43939015775880, -52.29998292239641, -64.16974420712729, -46.92765985464239, -60.91192496911521, -56.99539836354809, -55.31280343709037, -67.85061116175797, -49.59313106084930, -51.26117606150972, -50.20520476503064, -68.11262067397115, -60.00907540263483, -61.28979326912911, -60.78532612359061, -63.55888863433988, -65.83224954553405, -57.04318092695114, -62.27601062817367, -61.57386692545381, -46.49774533883657, -53.21696437570711, -55.48704406683728, -53.88268067446845, -54.28474458218987, -59.16733707051837, -52.23616188875616, -64.65348214266200, -55.43952739426679, -48.39526680576149, -47.43885524929092, -53.12336371796613, -47.78155369588303, -53.06000096546944, -50.17728341867199, -46.14749731038457, -69.84005635524851, -48.52544916378559, -48.31209187534131, -61.89346834172769, -52.42687265312647, -47.76779006853150, -50.60057459965100, -69.01586069382580, -68.10995141642621, -48.27179452839397, -57.37150216746025, -54.38442024015934, -61.39777325605118, -56.60770532475712, -63.63950362357492, -63.28840067040235, -48.64045128826957, -61.38589349507079, -53.05729352682290, -61.72823319145607, -55.60005974049413, -46.78965814370360, -67.29766604456466, -59.35379693142942, -48.11654510572903, -63.87819341258066, -59.41637140481463, -63.67983215841758, -48.62838006298267, -50.86788057598458, -68.82725294477795, -64.75897617862989, -52.17468917105704, -49.62188804702070, -47.19942839328228, -68.30682635649219, -56.13042717748861, -55.77892474783080, -47.94301893169410, -61.81471728984579, -48.46647879159157, -51.05606954132809, -46.68174450038606, -66.90100502649922, -54.15592590463166, -54.56002946443351, -64.76570601160884, -47.00853698378881, -53.85845829445269, -58.41190440671433, -61.95020609385605, -59.23651812547936, -53.74933772119775, -60.27759081348295, -69.67647292414019, -64.41718441849672, -46.40852737853533, -58.92874383893178, -57.70557679308189, -59.56404570235259, -51.06878169589726, -56.68380874079921, -50.56705839742372, -61.09259139523498, -51.35441608447042, -48.50090043339348, -54.99224267612109, -62.59150558178811, -62.95857919830319, -69.84188596949042, -66.34257429836428, -55.52290641392771, -63.41071752369851, -55.87135498496581, -64.15735261727896, -52.77704794693862, -51.69493480786874, -49.13073548810865, -56.25909420367420, -47.95405133763077, -60.59481863532666, -66.64589504516155, -69.26191945635993, -46.75121344372751, -55.47549366569390, -47.92758737269894, -63.41892913648262, -54.01602819676827, -52.69822503594173, -65.71939264945283, -66.89920128286903, -57.69143907345696, -62.75736153099258, -58.29083402020201, -55.61323417946567, -52.76119884566042, -56.65128762949829, -56.57196015236964, -47.13204991177245, -51.98724548599016, -50.17923739955071, -53.32240098799349, -69.26957061054436, -58.97064551959971, -57.35734654962953, -57.41529887202411, -54.83886622295703, -67.88526502294151, -69.25330696729768, -66.81506211543505, -47.13635016946488, -62.36080877082708, -53.52031211752529, -61.07124668598972, -63.70707335477466, -58.74485685407514, -60.33409250312576, -67.54653619435649, -51.70607674180013, -54.89152573597713, -63.55388056005044, -60.97558887482006, -59.44643052154531, -47.52022807992191, -47.82324161483216, -54.63149648069501, -61.87893170847710, -66.88521568112812, -51.25367618487661, -58.50260687352043, -67.36767437779464, -54.67018003205133, -55.05624711247092, -63.83084595187912, -50.84474144960949, -47.46844108850945, -50.97815268646684, -57.95184273852358, -58.47130112807218, -63.43156804480659, -56.74924650337952, -53.94896644878818, -46.28902454524707, -47.00456216039143, -50.20692794974699, -54.70261956481860, -68.16691298580780, -60.29632551783658, -61.62942728684878, -60.47616723957731, -64.78039897142668, -60.83028788612667, -55.18870418868822, -63.40621885661578, -66.37111424902602, -64.12964097880239, -69.70805613566414, -51.81779483511758, -47.92063811688686, -54.05428359890724, -58.23904914488800, -46.68806864284575, -57.39933304123244, -64.03896140630400, -62.49511802936468, -68.87443450142129, -64.29805262346041, -64.17736565783278, -68.55600164466968, -55.39640378699335, -48.10835245777852, -55.47725860727071, -65.39378940123511, -55.63140707394241, -60.73377440351314, -46.66960051266562, -52.63601691334154, -67.08911459847025, -54.44853224203918, -55.72452260558571, -67.73376145292733, -58.99763841424055, -57.01883098814233, -48.67417846237478, -60.30674513779846, -52.83849944412263, -46.64191709155105, -63.60321543480566, -60.85809646908260, -56.28858734282088, -57.24439571017983, -54.79284450754231, -49.16804485437694, -51.94883681847592, -64.41814794019432, -62.74537420630969, -57.93845921828598, -62.16746455159394, -58.46933496055804, -48.09521712265804, -64.14181503232477, -59.95527324704642, -47.34378911547128, -55.89048371655235, -64.13870006302915, -56.39842051049690, -54.19132885188112, -60.58251051509414, -46.94997684909299, -65.15188349813343, -60.19384925656922, -47.62356699316288, -55.61006540056005, -46.74908462714365, -58.71937455015131, -55.08231808533836, -55.94541598445534, -52.45156392576988, -62.26951533360806, -56.17064745433387, -66.02539379946977, -60.22816835147814, -65.27527807216723, -68.25140015165253, -58.86753511089245, -67.51706218801965, -48.00771235014304, -58.42486433781257, -64.75916771576978, -49.81076515094853, -53.18955021536517, -46.91642210240389, -63.69310543752822, -65.20588887872924, -67.26180597542431, -46.12265875252185, -49.61159014650513, -64.42389649167590, -58.39817689320390, -57.00168937003433, -64.79159755311257, -64.47326790631837, -47.82913721332297, -67.01177996503748, -57.66793566144430, -53.89762883856272, -53.41780160520837, -47.49421610341474, -61.73700791248427, -69.85446664295522, -61.92937211396516, -57.58006256698272, -51.46395484356007, -65.55499185377776, -55.31403015920258, -48.50004403734602, -48.98760857274331, -59.13656382787745, -64.65678747937207, -50.25786444810856, -56.23947640607387, -69.56720183944373, -65.09439797830186, -47.94757221333076, -48.52923275975844, -50.54849709528688, -67.95621807260487, -50.01901976227144, -58.43702055594055, -60.24256754730530, -49.59479212682569, -63.42013170855429, -52.36921066497059, -56.51891767503852, -52.22693044776236, -50.08707912972173, -65.13200268670573, -53.20515717053454, -56.75144033012403, -64.25582734896268, -57.66549049843479, -61.38453030480413, -46.07061358952845, -66.64788183247539, -66.11016426820225, -60.60583292329297, -66.52831291890774, -61.86136553504908, -47.72914828325273, -58.84974849602695, -68.31202882565725, -67.78198648469269, -48.10911756307541, -65.80313772212165, -64.65122463873914, -51.25980841800981, -62.50326124679013, -64.19914451406011, -62.90270542628310, -56.70971219569063, -65.39969503592356, -59.75616966163180, -57.57046950064586, -58.67407602138974, -54.13281235702182, -48.27750017194987, -54.00117391168569, -53.01896124668745, -65.28070257834428 };
    //vE.val = rtmp1;
    wE.val = (double*)malloc(NE * sizeof(double));
    sE.val = (double*)malloc(NE * sizeof(double));
    sEI.val = (double*)malloc(NE * sizeof(double));
    erE.val = (double*)malloc(NE * sizeof(double));
    edE.val = (double*)malloc(NE * sizeof(double));
    erEI.val = (double*)malloc(NE * sizeof(double));
    edEI.val = (double*)malloc(NE * sizeof(double));

    struct vec vI, wI, sI, sIE, erI, edI, erIE, edIE;
    vI.size = NI;
    wI.size = NI;
    sI.size = NI;
    sIE.size = NI;
    erI.size = NI;
    edI.size = NI;
    erIE.size = NI;
    edIE.size = NI;

    vI.val = (double *)malloc(NI * sizeof(double));
    //double rtmp2[] = { -68.9917112952736,-66.9611644329615,-58.7683461947669,-60.9212857291686,-63.748583060273,-66.483797956971,-66.5864700260394,-68.1348528660726,-66.1202421665246,-64.9478676210483,-61.6410164562847,-62.3715773280719,-61.7012356065501,-62.7983817904927,-63.3272725929005,-63.2613865851665,-68.9496641557385,-68.5327282863716,-58.7504498297275,-65.1210641446935,-59.5045028294206,-61.2043501924592,-61.8966175027292,-69.5544148485847,-61.6237784767964,-65.737161701314,-69.7540590670466,-69.5936443776445,-68.5598455239761,-60.3855057973246,-64.7052422533938,-60.9544792729002,-67.3763624029821,-62.3952701209615,-69.6684233422126,-67.7412185189089,-62.7529860085074,-64.4319624028862,-67.4013055510381,-62.3475892831148,-65.4330671659858,-60.6277897890857,-61.9279581135739,-61.4260762168969,-58.5723097199194,-67.5845122757376,-60.1850406934588,-67.9286072067368,-66.5699073284355,-68.1002709719589,-68.0562188235055,-62.9399997939391,-64.3067995270069,-60.0611643035628,-58.8572713918036,-61.1003639980764,-67.5383765638082,-62.8000838777776,-66.1078083897311,-66.8090915664673,-68.826268127445,-59.9311680014281,-67.9343757433489,-66.7967266329578,-59.3487423811132,-68.2523975416379,-62.3899734987997,-66.8885823984121,-62.8112366782452,-69.0977796077979,-64.1520620052574,-63.8122953704481,-68.8571063007969,-65.5039472658625,-65.7527831465534,-67.3799154927115,-61.029331544218,-62.8723100413356,-64.5424372906239,-63.028376549232,-68.7050568147296,-66.7302004818512,-67.4456828820801,-62.9873475959884,-66.6682243349132,-68.9392203775752,-66.5767783733947,-58.4977859780761,-64.7031784919005,-58.4132374908262,-64.933500135362,-61.0723541179937,-59.0762629965261,-66.6538842601076,-60.0759660020141,-59.8189300051926,-60.4986165381007,-62.4855318420131,-64.3097342534266,-68.1730618153334,-60.3394950121653,-62.0608418311024,-64.4935664072356,-62.2510126874961,-65.5842953225577,-66.5744528828386,-65.7187348678684,-63.7446056494075,-67.7347399457092,-62.5656938216145,-68.4497882367076,-61.1841007379451,-62.5378908068378,-63.0149078586825,-60.2618390986699,-69.205727902739,-65.5647038185264,-63.4273176561388,-60.0482837154426,-68.1174874134737,-62.3594946216219,-66.7722532456558,-61.6944513425141,-60.6142981702992,-63.300326941498,-64.3788293218588,-59.2331832265315,-65.2036702303059,-67.1181325711073,-64.0657456446892,-58.3383522642888,-65.0929575221979,-67.0055443833595,-60.0551763223725,-61.4483540603505,-66.0163016438588,-65.4477334629919,-61.5120133216398,-64.5955848462696,-63.5133026472482,-62.5596968433637,-62.6173685329633,-61.0559910594037,-62.1054153387089,-64.1173091432486,-67.0366781690367,-65.1851142018101,-67.824348064797,-63.627506856778,-63.2170878815496,-58.5418417133391,-66.3488141889778,-67.4795422638666,-58.8073158355988,-66.9553597997004,-59.109954468568,-64.0709461153611,-66.6537373046897,-61.7461507012379,-69.0850325259634 };
    //vI.val = rtmp2;
    wI.val = (double*)malloc(NI * sizeof(double));
    sI.val = (double*)malloc(NI * sizeof(double));
    sIE.val = (double*)malloc(NI * sizeof(double));
    erI.val = (double*)malloc(NI * sizeof(double));
    edI.val = (double*)malloc(NI * sizeof(double));
    erIE.val = (double*)malloc(NI * sizeof(double));
    edIE.val = (double*)malloc(NI * sizeof(double));


    double tmin, tmax;
    struct vec stsec;
    stsec.size = inps.on.size; //max size, then realloc
    stsec.val = (double*)malloc(stsec.size * sizeof(double));


    while (seqN <= T) {
        // reading the noise, to have it in a binary file @see scripts/SaveNoise.m 
        printf("[reading noise]\n");

        FILE* fp = NULL;
        if (seqN == 1)
            fp = fopen(ENOISE_BIN_PATH1, "rb");
        else
            fp = fopen(ENOISE_BIN_PATH2, "rb");
        if (fp == NULL) {
            perror("error while opening noises files");
            exit(1);
        }
        int numElements = fread(Enoise.val, sizeof(double), Enoise.size[0] * Enoise.size[1], fp);
        if (numElements < Enoise.size[0] * Enoise.size[1]) {
            if (ferror(fp)) {
                perror("error while reading noises files");
            }
            else {
                printf("[[warning: EOF before reading all the noises]]\n");
            }
            fclose(fp);
            exit(1);
        }
        fclose(fp);

        if (seqN == 1)
            fp = fopen(INOISE_BIN_PATH1, "rb");
        else
            fp = fopen(INOISE_BIN_PATH2, "rb");
        if (fp == NULL) {
            perror("error while opening noises files");
            exit(1);
        }
        numElements = fread(Inoise.val, sizeof(double), Inoise.size[0] * Inoise.size[1], fp);
        if (numElements < Inoise.size[0] * Inoise.size[1]) {
            if (ferror(fp)) {
                perror("error while reading noises files");
            }
            else {
                printf("[[warning: EOF before reading all the noises]]\n");
            }
            fclose(fp);
            exit(1);
        }
        fclose(fp);

        printf("integrating ODE\n");
        if (seqN == 1) {
            for (int i = 0; i < NE; i++) {
                vE.val[i] = ((double)rand() / (double)RAND_MAX) * (70 + p.VrE) - 70;
                wE.val[i] = p.aE * (vE.val[i] - p.ElE);
                sE.val[i] = 0;
                sEI.val[i] = 0;
                erE.val[i] = 0;
                edE.val[i] = 0;
                erEI.val[i] = 0;
                edEI.val[i] = 0;
            }

            for (int i = 0; i < NI; i++) {
                vI.val[i] = ((double)rand() / (double)RAND_MAX) * (70 + p.VrI) - 70;
                wI.val[i] = p.aI * (vI.val[i] - p.ElI);
                sI.val[i] = 0;
                sIE.val[i] = 0;
                erI.val[i] = 0;
                edI.val[i] = 0;
                erIE.val[i] = 0;
                edIE.val[i] = 0;
            }
        }
        else {
            tmin = (seqN - 1) * 1000 - 100;
            tmax = seqN * 1000 + 20;
            stsec.size = 0;
            for (int i = 0; i < inps.on.size; i++) {
                if (inps.on.val[i] >= tmin && inps.on.val[i] < tmax) {
                    if (i == inps.on.size - 1) // note that in the matlab version this part is wrong (so here)
                        break;
                    stsec.val[stsec.size] = inps.on.val[i];
                    stsec.size++;
                }
            }
            stsec.val = (double*)realloc(stsec.val, stsec.size * sizeof(double));

            struct matrix bmps;
            double* bt, * ebt;
            double ebt_max = 0;
            double bt_max = 0;
            bmps.size[0] = 100;
            bmps.size[1] = t.size;
            bmps.val = (double*)malloc(bmps.size[0] * bmps.size[1] * sizeof(double));
            bt = (double*)malloc(t.size * sizeof(double));
            ebt = (double*)malloc(t.size * sizeof(double));

            for (int j = 0; j < t.size; j++) {
                for (int i = 0; i < 100; i++) {
                    bmps.val[i * bmps.size[1] + j] = 0;
                }
                bt[j] = 0;
                ebt[j] = 0;
            }

            for (int i = 0; i < stsec.size; i++) {
                // inside each ripple event
                stsec.val[i] = stsec.val[i] - ((seqN - 1) * 1000);
                double rplton = stsec.val[i];
                double rpltoff = rplton + inps.length;
                int L = inps.length - inps.slp - 2;
                int L0 = 0; // inps.slp/inps.length
                int L1 = 1; // - L0

                double step = (double)1 / 99;
                int size = ceil((L1 - L0) / step) + 1;
                double* tbins, * tons, * toffs;
                tbins = (double*)malloc(size * sizeof(double));
                tons = (double*)malloc(size * sizeof(double));
                toffs = (double*)malloc(size * sizeof(double));
                for (int k = 0; k < size; k++) {
                    tbins[k] = (L0 + (k * step)) * L;
                    tons[k] = rplton + inps.slp + 2 + tbins[k]; // start the Ecells bumps after the I cells are inhibiting already
                    toffs[k] = tons[k] + (inps.length / 99);
                }
                for (int j = 0; j < bmps.size[1]; j++) {
                    for (int k = 0; k < bmps.size[0]; k++) {
                        bmps.val[k * bmps.size[1] + j] = bmps.val[k * bmps.size[1] + j] + (1 / (1 + exp((tons[k] - t.val[j]) / 1.5)) * 1 / (1 + exp((t.val[j] - toffs[k]) / 1.5)));
                    }
                    bt[j] = bt[j] + ((1 / (1 + exp((rplton - t.val[j]) / inps.slp))) * (1 / (1 + exp((t.val[j] - rpltoff) / inps.slp))));
                    ebt[j] = ebt[j] + ((1 / (1 + exp((rplton - t.val[j]) / (inps.slp / 2)))) * (1 / (1 + exp((t.val[j] - rpltoff) / (inps.slp / 2)))));
                    ebt_max = ebt_max > ebt[j] ? ebt_max : ebt[j];
                    bt_max = bt_max > bt[j] ? bt_max : bt[j];
                }

                free(tbins);
                free(tons);
                free(toffs);
            }

            struct matrix AEX, AIX;
            AEX.size[0] = NE;
            AEX.size[1] = bmps.size[1];
            AIX.size[0] = NI;
            AIX.size[1] = bmps.size[1];
            AEX.val = (double*)malloc(AEX.size[0] * AEX.size[1] * sizeof(double));
            AIX.val = (double*)malloc(AIX.size[0] * AIX.size[1] * sizeof(double));

            struct matrix tmp;
            tmp.size[0] = MX.size[0];
            tmp.size[1] = bmps.size[1];
            tmp.val = (double*)malloc(tmp.size[0] * tmp.size[1] * sizeof(double));

            productMatMat(&tmp, &MX, &bmps);

            for (int k = 0; k < AEX.size[0]; k++) {
                for (int j = 0; j < AEX.size[1]; j++) {
                    AEX.val[k * AEX.size[1] + j] = 5 * tmp.val[k * tmp.size[1] + j] + ebt[j];
                }
            }

            for (int k = 0; k < AIX.size[0]; k++) {
                for (int j = 0; j < AIX.size[1]; j++) {
                    AIX.val[k * AIX.size[1] + j] = bt[j];
                }
            }
            free(tmp.val);

            double Escale = ebt_max;
            double ff;
            ff = Escale > 0 ? p.jmpE / Escale : 0;
            for (int k = 0; k < Enoise.size[0]; k++) {
                for (int j = 0; j < Enoise.size[1]; j++) {
                    Enoise.val[k * Enoise.size[1] + j] = Enoise.val[k * Enoise.size[1] + j] + (ff * AEX.val[k * AEX.size[1] + j]);
                }
            }
            double Iscale = bt_max;
            double gg;
            gg = Iscale > 0 ? p.jmpI / Iscale : 0;
            for (int k = 0; k < Inoise.size[0]; k++) {
                for (int j = 0; j < Inoise.size[1]; j++) {
                    Inoise.val[k * Inoise.size[1] + j] = Inoise.val[k * Inoise.size[1] + j] + (gg * AIX.val[k * Inoise.size[1] + j]);
                }
            }

            int step = 1 / dt;
            Einptrace.size[1] += AEX.size[1] / step; // should be always 1000 (so += AEX.size[1] / step) or lower if dt is bigger? (so += step)
            Einptrace.val = (double*)realloc(Einptrace.val, Einptrace.size[0] * Einptrace.size[1] * sizeof(double));
            for (int k = 0; k < Einptrace.size[0]; k++) {
                for (int j = step + 1; j < Einptrace.size[1]; j++) {
                    Einptrace.val[k * Einptrace.size[1] + j] = ff * AEX.val[k * AEX.size[1] + ((j - step) * step)];
                }
            }
            Iinptrace.size[1] += AIX.size[1] / step; // should be always 1000 (so += AEX.size[1] / step) or lower if dt is bigger? (so += step)
            Iinptrace.val = (double*)realloc(Iinptrace.val, Iinptrace.size[0] * Iinptrace.size[1] * sizeof(double));
            for (int k = 0; k < Iinptrace.size[0]; k++) {
                for (int j = step + 1; j < Iinptrace.size[1]; j++) {
                    Iinptrace.val[k * Iinptrace.size[1] + j] = gg * AIX.val[k * AIX.size[1] + ((j - step) * step)];
                }
            }


            free(bmps.val);
            free(bt);
            free(ebt);
            free(AEX.val);
            free(AIX.val);

        }

        /* init gpu */

        cublasHandle_t handle; 
        cublasCreate(&handle);

        double* dev_GEE, * dev_GIE, * dev_GII, * dev_GEI;
        cudaMalloc((void**)&dev_GEE, GEE.size[0] * GEE.size[1] * sizeof(double));
        cudaMalloc((void**)&dev_GIE, GIE.size[0] * GIE.size[1] * sizeof(double));
        cudaMalloc((void**)&dev_GII, GII.size[0] * GII.size[1] * sizeof(double));
        cudaMalloc((void**)&dev_GEI, GEI.size[0] * GEI.size[1] * sizeof(double));
        cudaMemcpy(dev_GEE, GEE.val, GEE.size[0] * GEE.size[1] * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_GIE, GIE.val, GIE.size[0] * GIE.size[1] * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_GII, GII.val, GII.size[0] * GII.size[1] * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_GEI, GEI.val, GEI.size[0] * GEI.size[1] * sizeof(double), cudaMemcpyHostToDevice);

        double* dev_sE, * dev_sIE, * dev_sI, * dev_sEI;
        cudaMalloc((void**)&dev_sE, sE.size * sizeof(double));
        cudaMalloc((void**)&dev_sIE, sIE.size * sizeof(double));
        cudaMalloc((void**)&dev_sI, sI.size * sizeof(double));
        cudaMalloc((void**)&dev_sEI, sEI.size * sizeof(double));
        cudaMemcpy(dev_sE, sE.val, sE.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_sIE, sIE.val, sIE.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_sI, sI.val, sI.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_sEI, sEI.val, sEI.size * sizeof(double), cudaMemcpyHostToDevice);

        double* dev_tmp1, * dev_tmp2, * dev_tmp3, * dev_tmp4;
        cudaMalloc((void**)&dev_tmp1, NE * sizeof(double));
        cudaMalloc((void**)&dev_tmp2, NE * sizeof(double));
        cudaMalloc((void**)&dev_tmp3, NI * sizeof(double));
        cudaMalloc((void**)&dev_tmp4, NE * sizeof(double));

        double* dev_vE,* dev_vI;
        cudaMalloc((void**)&dev_vE, vE.size * sizeof(double));
        cudaMalloc((void**)&dev_vI, vI.size * sizeof(double));
        cudaMemcpy(dev_vE, vE.val, vE.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_vI, vI.val, vI.size * sizeof(double), cudaMemcpyHostToDevice);

        double *dev_lfp, * dev_vbarE, * dev_vbarI, * dev_vegE, *dev_vegI;
        cudaMalloc((void**)&dev_lfp, lfp.size * sizeof(double));
        cudaMalloc((void**)&dev_vbarE, lfp.size * sizeof(double));
        cudaMalloc((void**)&dev_vbarI, lfp.size * sizeof(double));
        cudaMalloc((void**)&dev_vegE, lfp.size * sizeof(double));
        cudaMalloc((void**)&dev_vegI, lfp.size * sizeof(double));
        cudaMemcpy(dev_lfp, lfp.val, lfp.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_vbarE, vbar.E, lfp.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_vbarI, vbar.I, lfp.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_vegE, veg.E, lfp.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_vegI, veg.I, lfp.size * sizeof(double), cudaMemcpyHostToDevice);

        double* dev_isynbarEtoE, * dev_isynbarItoE;
        cudaMalloc((void**)&dev_isynbarEtoE, isynbar.EtoE.size[0] * isynbar.EtoE.size[1] * sizeof(double));
        cudaMalloc((void**)&dev_isynbarItoE, isynbar.ItoE.size[0] * isynbar.ItoE.size[1] * sizeof(double));
        cudaMemcpy(dev_isynbarEtoE, isynbar.EtoE.val, isynbar.EtoE.size[0] * isynbar.EtoE.size[1] * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_isynbarItoE, isynbar.ItoE.val, isynbar.ItoE.size[0] * isynbar.ItoE.size[1] * sizeof(double), cudaMemcpyHostToDevice);

        double* dev_wE, * dev_wI;
        cudaMalloc((void**)&dev_wE, wE.size * sizeof(double));
        cudaMalloc((void**)&dev_wI, wI.size * sizeof(double));
        cudaMemcpy(dev_wE, wE.val, wE.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_wI, wI.val, wI.size * sizeof(double), cudaMemcpyHostToDevice);

        double* dev_Enoise, * dev_Inoise;
        cudaMalloc((void**)&dev_Enoise, Enoise.size[0] * Enoise.size[1] * sizeof(double));
        cudaMalloc((void**)&dev_Inoise, Inoise.size[0] * Inoise.size[1] * sizeof(double));
        cudaMemcpy(dev_Enoise, Enoise.val, Enoise.size[0] * Enoise.size[1] * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_Inoise, Inoise.val, Inoise.size[0] * Inoise.size[1] * sizeof(double), cudaMemcpyHostToDevice);

        double *dev_erE, * dev_edE, * dev_erEI, * dev_edEI, * dev_erI, * dev_edI, * dev_erIE, * dev_edIE;
        cudaMalloc((void**)&dev_erE, erE.size * sizeof(double));
        cudaMalloc((void**)&dev_edE, edE.size * sizeof(double));
        cudaMalloc((void**)&dev_erEI, erEI.size * sizeof(double));
        cudaMalloc((void**)&dev_edEI, edEI.size * sizeof(double));
        cudaMalloc((void**)&dev_erI, erI.size * sizeof(double));
        cudaMalloc((void**)&dev_edI, edI.size * sizeof(double));
        cudaMalloc((void**)&dev_erIE, erIE.size * sizeof(double));
        cudaMalloc((void**)&dev_edIE, edIE.size * sizeof(double));
        cudaMemcpy(dev_erE, erE.val, erE.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_edE, edE.val, edE.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_erEI, erEI.val, erEI.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_edEI, edEI.val, edEI.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_erI, erI.val, erI.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_edI, edI.val, edI.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_erIE, erIE.val, erIE.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_edIE, edIE.val, edIE.size * sizeof(double), cudaMemcpyHostToDevice);


        double* dev_tspEtimes, * dev_tspEcelln, * dev_tspItimes, * dev_tspIcelln;
        cudaMalloc((void**)&dev_tspEtimes, tspE.times.size * sizeof(double));
        cudaMalloc((void**)&dev_tspEcelln, tspE.celln.size * sizeof(double));
        cudaMalloc((void**)&dev_tspItimes, tspI.times.size * sizeof(double));
        cudaMalloc((void**)&dev_tspIcelln, tspI.celln.size * sizeof(double));
        cudaMemcpy(dev_tspEtimes, tspE.times.val, tspE.times.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_tspEcelln, tspE.celln.val, tspE.celln.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_tspItimes, tspI.times.val, tspI.times.size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_tspIcelln, tspI.celln.val, tspI.celln.size * sizeof(double), cudaMemcpyHostToDevice);

        int* dev_tspE_count, * dev_tspI_count;
        cudaMalloc((void**)&dev_tspE_count, sizeof(int));
        cudaMalloc((void**)&dev_tspI_count, sizeof(int));
        cudaMemcpy(dev_tspE_count, &tspE_count, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_tspI_count, &tspI_count, sizeof(int), cudaMemcpyHostToDevice);

        double* dev_t;
        cudaMalloc((void**)&dev_t, t.size * sizeof(double));
        cudaMemcpy(dev_t, t.val, t.size * sizeof(double), cudaMemcpyHostToDevice);


        dim3 dimBlockGen(1, 1, 1);
        dim3 dimGridGen(1, 1, 1);
        dim3 dimBlockE(32, 1, 1);
        dim3 dimGridE(25, 1, 1);
        dim3 dimBlockI(32, 1, 1);
        dim3 dimGridI(5, 1, 1);

        const double dev_alpha = 1.0;
        const double dev_beta = 0.0;

        for (int idt = 1; idt < t.size; idt++) {
            // check the GEE(..) sizes and the transpositions.
            cublasDgemv(handle, CUBLAS_OP_T, NE, NE, &dev_alpha, dev_GEE, NE, dev_sE, 1, &dev_beta, dev_tmp1, 1);
            cublasDgemv(handle, CUBLAS_OP_T, NI, NE, &dev_alpha, dev_GIE, NI, dev_sIE, 1, &dev_beta, dev_tmp2, 1);
            cublasDgemv(handle, CUBLAS_OP_T, NI, NI, &dev_alpha, dev_GII, NI, dev_sI, 1, &dev_beta, dev_tmp3, 1);
            cublasDgemv(handle, CUBLAS_OP_T, NI, NE, &dev_alpha, dev_GEI, NI, dev_sEI, 1, &dev_beta, dev_tmp4, 1);
            //cudaDeviceSynchronize();



            /* do on gpu */
            kernelGen << <dimGridGen, dimBlockGen >> > (NE, NI,dev_sE, dev_sIE, dev_sI, dev_sEI, dev_GEE, dev_GIE, dev_GII, dev_GEI, dev_tmp1, dev_tmp2, dev_tmp3, dev_tmp4, idt, seqN, dev_vE, dev_vI, dev_lfp, dev_vbarE, dev_vbarI, dev_vegE, dev_vegI, Eeg, Ieg, opt.storecurrs, dev_isynbarEtoE, dev_isynbarItoE, lfp.size, p.VrevE, p.VrevI, p.glE, p.ElE, p.slpE, p.VtE, p.CE, p.aE, p.twE, p.VrE, p.bE, p.glI, p.ElI, p.slpI, p.VtI, p.CI, p.aI, p.twI, p.VrI, p.bI, dev_wE, dev_wI, dev_Enoise, dev_Inoise, Enoise.size[1], Inoise.size[1], dt, dev_erE, dev_edE, dev_erEI, dev_edEI, dev_erI, dev_edI, dev_erIE, dev_edIE, fdE, frE, fdEI, frEI, pvsE, pvsEI, fdI, frI, fdIE, frIE, pvsI, pvsIE, dev_tspEtimes, dev_tspEcelln, dev_tspItimes, dev_tspIcelln, dev_tspE_count, dev_tspI_count, dev_t);
            kernelE << <dimGridE, dimBlockE >> > (NE, NI,dev_sE, dev_sIE, dev_sI, dev_sEI, dev_GEE, dev_GIE, dev_GII, dev_GEI, dev_tmp1, dev_tmp2, dev_tmp3, dev_tmp4, idt, seqN, dev_vE, dev_vI, dev_lfp, dev_vbarE, dev_vbarI, dev_vegE, dev_vegI, Eeg, Ieg, opt.storecurrs, dev_isynbarEtoE, dev_isynbarItoE, lfp.size, p.VrevE, p.VrevI, p.glE, p.ElE, p.slpE, p.VtE, p.CE, p.aE, p.twE, p.VrE, p.bE, p.glI, p.ElI, p.slpI, p.VtI, p.CI, p.aI, p.twI, p.VrI, p.bI, dev_wE, dev_wI, dev_Enoise, dev_Inoise, Enoise.size[1], Inoise.size[1], dt, dev_erE, dev_edE, dev_erEI, dev_edEI, dev_erI, dev_edI, dev_erIE, dev_edIE, fdE, frE, fdEI, frEI, pvsE, pvsEI, fdI, frI, fdIE, frIE, pvsI, pvsIE, dev_tspEtimes, dev_tspEcelln, dev_tspItimes, dev_tspIcelln, dev_tspE_count, dev_tspI_count, dev_t);
            kernelI << <dimGridI, dimBlockI >> > (NE, NI,dev_sE, dev_sIE, dev_sI, dev_sEI, dev_GEE, dev_GIE, dev_GII, dev_GEI, dev_tmp1, dev_tmp2, dev_tmp3, dev_tmp4, idt, seqN, dev_vE, dev_vI, dev_lfp, dev_vbarE, dev_vbarI, dev_vegE, dev_vegI, Eeg, Ieg, opt.storecurrs, dev_isynbarEtoE, dev_isynbarItoE, lfp.size, p.VrevE, p.VrevI, p.glE, p.ElE, p.slpE, p.VtE, p.CE, p.aE, p.twE, p.VrE, p.bE, p.glI, p.ElI, p.slpI, p.VtI, p.CI, p.aI, p.twI, p.VrI, p.bI, dev_wE, dev_wI, dev_Enoise, dev_Inoise, Enoise.size[1], Inoise.size[1], dt, dev_erE, dev_edE, dev_erEI, dev_edEI, dev_erI, dev_edI, dev_erIE, dev_edIE, fdE, frE, fdEI, frEI, pvsE, pvsEI, fdI, frI, fdIE, frIE, pvsI, pvsIE, dev_tspEtimes, dev_tspEcelln, dev_tspItimes, dev_tspIcelln, dev_tspE_count, dev_tspI_count, dev_t);
        }

        /* get back from gpu */

        // GEE(...) don't change
        cudaMemcpy(sE.val, dev_sE, sE.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(sIE.val, dev_sIE, sIE.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(sI.val, dev_sI, sI.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(sEI.val, dev_sEI, sEI.size * sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(vE.val, dev_vE, vE.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(vI.val, dev_vI, vI.size * sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(lfp.val, dev_lfp, lfp.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(vbar.E, dev_vbarE, lfp.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(vbar.I, dev_vbarI, lfp.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(veg.E, dev_vegE, lfp.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(veg.I, dev_vegI, lfp.size * sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(isynbar.EtoE.val, dev_isynbarEtoE, isynbar.EtoE.size[0] * isynbar.EtoE.size[1] * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(isynbar.ItoE.val, dev_isynbarItoE, isynbar.ItoE.size[0] * isynbar.ItoE.size[1] * sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(wE.val, dev_wE, wE.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(wI.val, dev_wI, wI.size * sizeof(double), cudaMemcpyDeviceToHost);


        // Enoise and Inoise don't change
        cudaMemcpy(erE.val, dev_erE, erE.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(edE.val, dev_edE, edE.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(erEI.val, dev_erEI, erEI.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(edEI.val, dev_edEI, edEI.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(erI.val, dev_erI, erI.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(edI.val, dev_edI, edI.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(erIE.val, dev_erIE, erIE.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(edIE.val, dev_edIE, edIE.size * sizeof(double), cudaMemcpyDeviceToHost);

        cudaMemcpy(tspE.times.val, dev_tspEtimes, tspE.times.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(tspE.celln.val, dev_tspEcelln, tspE.celln.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(tspI.times.val, dev_tspItimes, tspI.times.size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(tspI.celln.val, dev_tspIcelln, tspI.celln.size * sizeof(double), cudaMemcpyDeviceToHost);


        cudaMemcpy(&tspE_count, dev_tspE_count, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&tspI_count, dev_tspI_count, sizeof(int), cudaMemcpyDeviceToHost);


        cudaMemcpy(t.val, dev_t, t.size * sizeof(double), cudaMemcpyDeviceToHost);

        /* destroy the trash */
        cublasDestroy(handle);

        cudaFree(dev_tmp1);
        cudaFree(dev_tmp2);
        cudaFree(dev_tmp3);
        cudaFree(dev_tmp4);

        cudaFree(dev_GEE);
        cudaFree(dev_GII);
        cudaFree(dev_GEI);
        cudaFree(dev_GIE);

        cudaFree(dev_sE);
        cudaFree(dev_sI);
        cudaFree(dev_sEI);
        cudaFree(dev_sIE);

        cudaFree(dev_vE);
        cudaFree(dev_vI);

        cudaFree(dev_lfp);
        cudaFree(dev_vbarE);
        cudaFree(dev_vbarI);
        cudaFree(dev_vegE);
        cudaFree(dev_vegI);

        cudaFree(dev_isynbarEtoE);
        cudaFree(dev_isynbarItoE);

        cudaFree(dev_wE);
        cudaFree(dev_wI);

        cudaFree(dev_Enoise);
        cudaFree(dev_Inoise);

        cudaFree(dev_erE);
        cudaFree(dev_edE);
        cudaFree(dev_erEI);
        cudaFree(dev_edEI);
        cudaFree(dev_erI);
        cudaFree(dev_edI);
        cudaFree(dev_erIE);
        cudaFree(dev_edIE);

        cudaFree(dev_tspEtimes);
        cudaFree(dev_tspEcelln);
        cudaFree(dev_tspItimes);
        cudaFree(dev_tspIcelln);

        cudaFree(dev_tspE_count);
        cudaFree(dev_tspI_count);

        cudaFree(dev_t);

        seqN++;
        clock_t toc = clock();
        printf("elapsed time is %.2lf seconds.\n", (double)(toc - tic) / CLOCKS_PER_SEC);
    }

    tspE.times.size = tspE_count;
    tspE.celln.size = tspE_count;
    tspE.times.val = (double *)realloc(tspE.times.val, tspE.times.size * sizeof(double));
    tspE.celln.val = (double *)realloc(tspE.celln.val, tspE.celln.size * sizeof(double));

    tspI.times.size = tspI_count;
    tspI.celln.size = tspI_count;
    tspI.times.val = (double*)realloc(tspI.times.val, tspI.times.size * sizeof(double));
    tspI.celln.val = (double*)realloc(tspI.celln.val, tspI.celln.size * sizeof(double));

    inp.Etrace = Einptrace;
    inp.Itrace = Iinptrace;

    /* free */

    free(MX.val);  
    free(GEE.val);
    free(GII.val);
    
    free(Enoise.val);
    free(Inoise.val);
    
    free(vE.val); 
    free(wE.val); 
    free(sE.val); 
    free(sEI.val); 
    free(erE.val); 
    free(edE.val); 
    free(erEI.val); 
    free(edEI.val);
    free(vI.val);
    free(wI.val);
    free(sI.val);
    free(sIE.val);
    free(erI.val);
    free(edI.val);
    free(erIE.val);
    free(edIE.val);
    free(stsec.val);

    //free(conn.EtoE.val);
    //free(conn.EtoI.val);
    //free(conn.ItoE.val);
    //free(conn.ItoI.val);
    //free(vbar.E);
    //free(vbar.I);
    //free(isynbar.EtoE.val);
    //free(isynbar.ItoE.val);
    //free(seqs);

    //save results
    save(&veg, &lfp, &tspE, &tspI, &inp, &inps, T, NE, NI);

    return;
}