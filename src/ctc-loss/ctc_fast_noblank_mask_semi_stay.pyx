
from libc cimport math
import cython
import numpy as np
cimport numpy as np
np.seterr(divide='raise',invalid='raise')

ctypedef np.float64_t f_t

# Turn off bounds checking, negative indexing
@cython.boundscheck(False)
@cython.wraparound(False)
def ctc_loss(double[::1,:] params not None, 
        int[::1] seq not None, 
        int[::1] semi_labels not None, ### semi_supervised label
        int[::1] mask_S not None, int[::1] mask_E not None, ### MASK
        double[::1] stay_prob_F not None, double[::1] stay_prob_B not None, # forward and backward stay prob
        unsigned int blank=0):
    """
    CTC loss function.
    params - n x m matrix of n-D probability distributions over m frames. Must
    be in Fortran order in memory.
    seq - sequence of phone id's for given example.
    Returns objective and gradient.
    """

    cdef unsigned int seqLen = seq.shape[0] # Length of label sequence (# phones)
    cdef unsigned int numphones = params.shape[0] # Number of labels
    cdef unsigned int L = 2*seqLen + 1 # Length of label sequence with blanks
    cdef unsigned int T = params.shape[1] # Length of utterance (time)

    cdef double[::1,:] alphas = np.zeros((L,T), dtype=np.double, order='F')
    cdef double[::1,:] betas = np.zeros((L,T), dtype=np.double, order='F')
    cdef double[::1,:] ab = np.empty((L,T), dtype=np.double, order='F')
    cdef np.ndarray[f_t, ndim=2] grad = np.zeros((numphones,T), 
                            dtype=np.double, order='F')
    cdef double[::1,:] grad_v = grad
    cdef double[::1] absum = np.empty(T, dtype=np.double)

    cdef unsigned int start, end
    cdef unsigned int t, s, l
    cdef double c, llForward, llBackward, llDiff, tmp


    N = numphones - 1 # number of actions, minus one for blank
    # print N,stay_prob_F[0],stay_prob_B[10]  

    try:
        # Initialize alphas and forward pass 
    
        # noblank
        # alphas[0,0] = params[blank,0]
        alphas[0,0] = 0
              
        alphas[1,0] = params[seq[0],0]
        c = alphas[0,0] + alphas[1,0]
        alphas[0,0] = alphas[0,0] / c
        alphas[1,0] = alphas[1,0] / c
        llForward = math.log(c)
        for t in xrange(1,T):
            start = 2*(T-t)
            if L <= start:
                start = 0
            else:
                start = L-start
            end = min(2*t+2,L)
            for s in xrange(start,L):
                l = (s-1)/2
                
                ##### SEMI #####
                if semi_labels[t] > 0:
                    # have semisupervised label
                    if s%2 == 0:
                        # skip if blank, because we have gnd label
                        continue
                    elif seq[l] != semi_labels[t]:
                        # skip if the corresponding label of s is not the same as supervision
                        continue
                ##### SEMI #####
                
                ##### MASK #####
                if s%2 == 0:
                    # blank
                    if l not in range(mask_S[t], mask_E[t]):
                        # as blank will reduce to the previous l, also check l+1
                        if l+1 not in range(mask_S[t], mask_E[t]):
                            continue
                else:
                    # not blank
                    if l not in range(mask_S[t], mask_E[t]):
                        continue 
                ##### MASK #####   
                        
                ##### PARAMS_STAY & PARAMS_MOVE #####
                if s%2 != 0: # not needed for blank
                    #params_stay = params[seq[l],t] 
                    params_ori = params[seq[l],t] 
                    
                    p_stay = stay_prob_F[t] # change for backward pass
                    p_move = (1 - p_stay) / (N - 1) # (N-1)*p_change + p_stay = 1
                        
                    params_stay = params_ori * p_stay
                    div = ( params_ori * p_stay ) + ( (1 - params_ori) * p_move ) # sum     
                    params_stay = params_stay / div # normalize
                    
                    if s > 1: # has previous
                        s_pre = s-2
                        l_pre = (s_pre-1)/2
                        params_pre = params[seq[l_pre],t] 
                        
                        params_move = params_ori * p_move
                        div = ( params_pre * p_stay ) + ( (1-params_pre) * p_move )
                        params_move = params_move / div # normalize                        
                        
                    #if t%1000 == 0 and s > 1:
                    #    print t,s,params_stay, params_move
                ##### PARAMS_STAY & PARAMS_MOVE #####                 
                
                # blank
                # noblank, but still keep so no redesign for other parts of the code
                if s%2 == 0:
                    # noblank, always zero
                    #if s==0:
                    #    alphas[s,t] = alphas[s,t-1] * params[blank,t]
                    #else:
                    #    alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[blank,t]
                    alphas[s,t] = 0    
                        
                # same label twice
                # no same label twice for nobalnk
                # elif s == 1 or seq[l] == seq[l-1]:
                elif s == 1:
                    # noblank
                    # alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[seq[l],t]   
                
                    # alphas[s,t] = alphas[s,t-1] * params[seq[l],t]
                
                    alphas[s,t] = alphas[s,t-1] * params_stay
                else:
                    # noblank
                    #alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1] + alphas[s-2,t-1]) \
                    #              * params[seq[l],t]

                    #alphas[s,t] = (alphas[s,t-1] + alphas[s-2,t-1]) * params[seq[l],t]  
                    alphas[s,t] = alphas[s,t-1] * params_stay \
                                    + alphas[s-2,t-1] * params_move                                
                
            # normalize at current time (prevent underflow)
            c = 0.0
            for s in xrange(start,end):
                c += alphas[s,t]
            for s in xrange(start,end):
                alphas[s,t] = alphas[s,t] / c
            llForward += math.log(c)

        # Initialize betas and backwards pass
        # noblank
        # betas[L-1,T-1] = params[blank,T-1]
        betas[L-1,T-1] = 0
        betas[L-2,T-1] = params[seq[seqLen-1],T-1]
        c = betas[L-1,T-1] + betas[L-2,T-1]
        betas[L-1,T-1] = betas[L-1,T-1] / c
        betas[L-2,T-1] = betas[L-2,T-1] / c
        llBackward = math.log(c)
        for t in xrange(T-1,0,-1):
            t = t-1
            start = 2*(T-t)
            if L <= start:
                start = 0
            else:
                start = L-start
            end = min(2*t+2,L)
            for s in xrange(end,0,-1):
                s = s-1
                l = (s-1)/2

                ##### SEMI #####
                if semi_labels[t] > 0:
                    # have semisupervised label
                    if s%2 == 0:
                        # skip if blank, because we have gnd label
                        continue
                    elif seq[l] != semi_labels[t]:
                        # skip if the corresponding label of s is not the same as supervision
                        continue
                ##### SEMI #####
                        
                ##### MASK #####
                if s%2 == 0:
                    # blank
                    if l not in range(mask_S[t], mask_E[t]):
                        # as blank will reduce to the previous l, also check l+1
                        if l+1 not in range(mask_S[t], mask_E[t]):
                            continue
                else:
                    # not blank
                    if l not in range(mask_S[t], mask_E[t]):
                        continue 
                ##### MASK #####                   
                
                # TODO use stay_prob_B
                # params_stay = params[seq[l],t] 
                # params_move = params[seq[l],t] 
                                
                
                
                ##### PARAMS_STAY & PARAMS_MOVE #####
                if s%2 != 0: # not needed for blank
                    params_ori = params[seq[l],t]
                    
                    p_stay = stay_prob_B[t] # diff
                    p_move = (1 - p_stay) / (N - 1) # (N-1)*p_change + p_stay = 1

                    params_stay = params_ori * p_stay
                    div = ( params_ori * p_stay ) + ( (1-params_ori) * p_move ) # sum
                    params_stay = params_stay / div # normalize
                    
                    if s < L-2: #has next
                        s_next = s+2
                        l_next = (s_next-1)/2
                        params_next = params[seq[l_next],t]
                        
                        params_move = params_ori * p_move
                        div = (params_next * p_stay) + ( (1-params_next) * p_move )
                        params_move = params_move / div

                    #if t%1000 == 0 and s < L-2:
                    #    print t,s,params_stay, params_move
                ##### PARAMS_STAY & PARAMS_MOVE #####     
                
                # blank
                if s%2 == 0:
                    # noblank, so all zero
                    #if s == L-1:
                    #    betas[s,t] = betas[s,t+1] * params[blank,t]
                    #else:
                    #    betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * params[blank,t]
                    betas[s,t] = 0
                # noblank => no same label twice
                # same label twice
                #elif s == L-2 or seq[l] == seq[l+1]:
                elif s == L-2:
                    # noblank => betas[s+1, t+1] = 0
                    # betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * params[seq[l],t]
                               
                
                    #betas[s,t] = betas[s,t+1] * params[seq[l],t]
                    betas[s,t] = betas[s,t+1] * params_stay
                    
                else:
                    # noblank => betas[s+1, t+1] = 0
                    #betas[s,t] = (betas[s,t+1] + betas[s+1,t+1] + betas[s+2,t+1]) \
                    #             * params[seq[l],t]
                
                
                    #betas[s,t] = (betas[s,t+1] + betas[s+2,t+1]) * params[seq[l],t]  
                    betas[s,t] = betas[s,t+1] * params_stay \
                                    + betas[s+2,t+1] * params_move                              

            c = 0.0
            for s in xrange(start,end):
                c += betas[s,t]
            for s in xrange(start,end):
                betas[s,t] = betas[s,t] / c
            llBackward += math.log(c)

        # Compute gradient with respect to unnormalized input parameters
        for t in xrange(T):
            for s in xrange(L):
                ab[s,t] = alphas[s,t]*betas[s,t]
        for s in xrange(L):
            # blank
            if s%2 == 0:
                for t in xrange(T):
                    # noblank: will be zero => will do nothing
                    grad_v[blank,t] += ab[s,t]
                    # noblank: will be zero => will do nothing
                    if ab[s,t] != 0:
                        ab[s,t] = ab[s,t]/params[blank,t]
            else:
                for t in xrange(T):
                    grad_v[seq[(s-1)/2],t] += ab[s,t]
                    if ab[s,t] != 0:
                        ab[s,t] = ab[s,t]/(params[seq[(s-1)/2],t]) 

        for t in xrange(T):
            absum[t] = 0
            for s in xrange(L):
                absum[t] += ab[s,t]

        # grad = params - grad / (params * absum)
        for t in xrange(T):
            for s in xrange(numphones):
                tmp = (params[s,t]*absum[t])
                if tmp > 0:
                    grad_v[s,t] = params[s,t] - grad_v[s,t] / tmp 
                else:
                    grad_v[s,t] = params[s,t]

    except (FloatingPointError,ZeroDivisionError) as e:
        print e.message
        return -llForward,grad,True


    return -llForward,grad,False

def decode_best_path(double[::1,:] probs not None, unsigned int blank=0):
    """
    Computes best path given sequence of probability distributions per frame.
    Simply chooses most likely label at each timestep then collapses result to
    remove blanks and repeats.
    Optionally computes edit distance between reference transcription
    and best path if reference provided.
    """

    # Compute best path
    cdef unsigned int T = probs.shape[1]
    cdef long [::1] best_path = np.argmax(probs,axis=0)

    # Collapse phone string
    cdef unsigned int i, b
    hyp = []
    align = []
    for i in xrange(T):
        b = best_path[i]
        # ignore blanks
        if b == blank:
            continue
        # FIXME ignore some special characters
        # noise, laughter, vocalized-noise
        if b == 1 or b == 2 or b == 8:
            continue
        # ignore repeats
        elif i != 0 and  b == best_path[i-1]:
            align[-1] = i
            continue
        else:
            hyp.append(b)
            align.append(i)
    return hyp, align

