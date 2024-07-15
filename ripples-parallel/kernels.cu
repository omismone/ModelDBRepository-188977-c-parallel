#include "omislib.cuh"

__global__ void kernelGen(int NE, int NI, double* dev_tmp1, double* dev_tmp2, int idt, int seqN, double* dev_vE, double* dev_vI, double* dev_lfp, double* dev_vbarE, double* dev_vbarI, double* dev_vegE, double* dev_vegI, int Eeg, int Ieg, int opt_storecurrs, double* dev_isynbarEtoE, double* dev_isynbarItoE, int lfp_size, double pVrevE, double pVrevI, double* dev_tspEtimes, double* dev_tspEcelln, double* dev_tspItimes, double* dev_tspIcelln, int* dev_tspE_count, int* dev_tspI_count, double* dev_t) {
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

__global__ void kernelE(int NE, double* dev_sE, double* dev_sEI, double* dev_tmp1, double* dev_tmp2, int idt, double* dev_vE, double pVrevE, double pVrevI, double pglE, double pElE, double pslpE, double pVtE, double pCE, double paE, double ptwE, double pVrE, double pbE, double* dev_wE, double* dev_Enoise, int dev_Enoise_size_1, double dev_dt, double* dev_erE, double* dev_edE, double* dev_erEI, double* dev_edEI, double dev_fdE, double dev_frE, double dev_fdEI, double dev_frEI, double dev_pvsE, double dev_pvsEI) {
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    if (idx >= NE)
        return;
    double Isyn, Iapp, Iion, dvdt, dwdt, wEst, vEst;
    vEst = dev_vE[idx];
    wEst = dev_wE[idx];
    Isyn = (dev_tmp1[idx] * (dev_vE[idx] - pVrevE)) + (dev_tmp2[idx] * (dev_vE[idx] - pVrevI));
    Iapp = dev_Enoise[idx * dev_Enoise_size_1 + (idt - 1)];
    Iion = ((-1) * pglE * (dev_vE[idx] - pElE)) + (pglE * pslpE * exp((dev_vE[idx] - pVtE) / pslpE)) - (dev_wE[idx]);
    dvdt = ((Iapp + (Iion)-Isyn) / pCE);
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

__global__ void kernelI(int NI, double* dev_sIE, double* dev_sI, double* dev_tmp3, double* dev_tmp4, int idt, double* dev_vI, double pVrevE, double pVrevI, double pglI, double pElI, double pslpI, double pVtI, double pCI, double paI, double ptwI, double pVrI, double pbI, double* dev_wI, double* dev_Inoise, int dev_Inoise_size_1, double dev_dt, double* dev_erI, double* dev_edI, double* dev_erIE, double* dev_edIE, double dev_fdI, double dev_frI, double dev_fdIE, double dev_frIE, double dev_pvsI, double dev_pvsIE) {
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    if (idx >= NI)
        return;
    double Isyn, Iapp, Iion, dvdt, dwdt, wIst, vIst;
    vIst = dev_vI[idx];
    wIst = dev_wI[idx];
    Isyn = (dev_tmp3[idx] * (dev_vI[idx] - pVrevI)) + (dev_tmp4[idx] * (dev_vI[idx] - pVrevE));
    Iapp = dev_Inoise[idx * dev_Inoise_size_1 + (idt - 1)]; 
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