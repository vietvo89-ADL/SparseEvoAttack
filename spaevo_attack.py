import torch
import numpy as np
from utils_se import l0

# main attack
class SpaEvoAtt():
    def __init__(self,
                model,
                pop_init='grid_rand',
                n = 16,
                pop_size=10,
                #grid_size=16,
                cr=0.9,
                mu=0.01,
                seed = None,
                flag=True):

        self.model = model
        self.pop_init = pop_init
        self.n_pix = n # if uni_rand is used
        self.pop_size = pop_size
        self.grid_size = n # if grid_rand is used
        self.cr = cr
        self.mu = mu
        self.seed = seed
        self.flag = flag

    def convert1D_to_2D(self,idx,wi):
        c1 = idx //wi
        c2 = idx - c1 * wi
        return c1, c2

    def convert2D_to_1D(self,x,y,wi):
        outp = x*wi + y
        return outp

    def segment_convert_pixID(self,n, wi,gsize):
        c1 = n // wi
        c2 = n - c1 * wi
        c1 *= gsize
        c2 *= gsize
        return c1,c2

    def masking(self,oimg,timg):
        xo = torch.abs(oimg-timg)
        d = torch.zeros(xo.shape[2],xo.shape[3]).bool().cuda()
        for i in range (xo.shape[1]):
            tmp = (xo[0,i]>0.).bool().cuda()
            d = tmp | d # "or" => + ; |
        
        wi = oimg.shape[2]
        p = np.where(d.int().cpu().numpy() == 1) # oimg -> reference;'0' => "same as oimg" '1' => 'same as timg'
        out = self.convert2D_to_1D(p[0],p[1],wi)

        return out # output pixel coordinates have value same as 'timg'

    def grid_rand(self,oimg,timg,olabel,tlabel):
        
        if self.seed != None:
            np.random.seed(self.seed)

        nqry = 0
        wi = oimg.shape[2]
        he = oimg.shape[3]
        fit = torch.zeros(self.pop_size) + np.inf
        pop = []
        p1 = np.zeros(wi*he).astype(int)
        idxs = self.masking(oimg,timg) #[x for x in range(wi * he)]
        p1[idxs] = 1
        #print('after sp att:',p1.sum())
    
        for i in range(self.pop_size):
            found = False
            gsize = self.grid_size
            w = oimg.shape[2]//self.grid_size
            h = oimg.shape[3]//self.grid_size
            n = w*h
            cnt = 0
            while True:
                p = p1.copy()
                for j in range(n): #n = number of perturbed pixels
                    c1,c2 = self.segment_convert_pixID(j,w,gsize)
                    x = np.random.randint(c1,c1+gsize,size=1)
                    y = np.random.randint(c2,c2+gsize,size=1)
                    k = self.convert2D_to_1D(x[0],y[0],wi)
                    p[k] = 0
                
                nqry += 1
                fitness = self.feval(p,oimg,timg,olabel,tlabel)
                if fitness < fit[i]:
                    pop.append(p)
                    fit[i] = fitness
                    break
                else:
                    cnt += 1
                if (cnt == self.pop_size)and(gsize<=wi):
                    cnt = 0
                    gsize *= 2
                    w = oimg.shape[2]//gsize
                    h = oimg.shape[3]//gsize
                    n = w*h    
            
        return pop,nqry,fit

    def uni_rand(self,oimg,timg,olabel,tlabel):
    
        if self.seed != None:
            np.random.seed(self.seed)

        terminate = False
        nqry = 0
        wi = oimg.shape[2]
        he = oimg.shape[3]
        
        fit = torch.zeros(self.pop_size) + np.inf
        pop = []

        p1 = np.zeros(wi*he).astype(int)
        idxs = self.masking(oimg,timg) #[x for x in range(wi * he)]
        p1[idxs] = 1
        #print('after sp att:',p1.sum())
        
        if p1.sum()<self.n_pix:
            self.n_pix = p1.sum()        

        for i in range(self.pop_size):
            n = self.n_pix
            cnt = 0
            j = 0
            while True:
                p = p1.copy()
                idx = np.random.choice(idxs, n, replace = False)
                p[idx] = 0
                nqry += 1
                fitness = self.feval(p,oimg,timg,olabel,tlabel)
                    
                if fitness < fit[i]:
                    pop.append(p)
                    fit[i] = fitness
                    break
                elif (n>1):
                    n -= 1
                elif (n == 1):
                    #for j in range(len(idxs)):
                    while j < len(idxs):
                        p[idxs[j]] = 0
                        nqry += 1
                        fitness = self.feval(p,oimg,timg,olabel,tlabel)

                        if fitness < fit[i]:
                            pop.append(p)
                            fit[i] = fitness
                            break
                        else:
                            j += 1
                    
                    #if (j==len(idxs)-1):
                    #    terminate = True
                    break

            #if terminate:
            if (j==len(idxs)-1):
                break
                
        if len(pop)<self.pop_size:
            for i in range(len(pop),self.pop_size):
                pop.append(p1)

        return pop,nqry,fit

    def recombine(self,p0,p1,p2):

        cross_points = np.random.rand(len(p1)) < self.cr # uniform random
        if not np.any(cross_points):
            cross_points[np.random.randint(0, len(p1))] = True
        trial = np.where(cross_points, p1, p2).astype(int)
        trial = np.logical_and(p0,trial).astype(int) 
        return trial

    def mutate(self,p):

        outp = p.copy()
        if p.sum() != 0:
            one = np.where(outp == 1)
            n_pix = int(len(one[0])*self.mu)
            if n_pix == 0:
                n_pix = 1
            idx = np.random.choice(one[0],n_pix,replace=False)
            outp[idx] = 0

        return outp

    def modify(self,pop,oimg,timg):
        wi = oimg.shape[2]
        img = timg.clone()
        p = np.where(pop == 0)
        c1,c2 = self.convert1D_to_2D(p[0],wi)
        img[:,:,c1,c2] = oimg[:,:,c1,c2]
        return img

    def feval(self,pop,oimg,timg,olabel,tlabel):

        xp = self.modify(pop,oimg,timg)
        l2 = torch.norm(oimg - xp).cpu().numpy()
        pred_label = self.model.predict_label(xp)

        if self.flag == True:
            if pred_label == tlabel:
                lc = 0
            else:
                lc = np.inf
        else:
            if pred_label != olabel:
                lc = 0
            else:
                lc = np.inf

        outp = l2 + lc
        return outp 

    def selection(self,x1,f1,x2,f2):

        xo = x1.copy()
        fo = f1
        if f2<f1:
            fo = f2
            xo = x2

        return xo,fo

    def evo_perturb(self,oimg,timg,olabel,tlabel,max_query=1000):

        # 0. variable init
        if self.seed != None:
            np.random.seed(self.seed)

        D = np.zeros(max_query+1000)
        #nq = 0
        wi = oimg.shape[3]
        he = oimg.shape[2]
        n_dims = wi * he
        # 1. population init
        idxs = self.masking(oimg,timg) #[x for x in range(wi * he)]
        if len(idxs)>1: # more than 1 diff pixel
            if self.pop_init == 'uni_rand':
                pop, nqry,fitness = self.uni_rand(oimg,timg,olabel,tlabel)
            else: # grid_rand
                pop, nqry,fitness = self.grid_rand(oimg,timg,olabel,tlabel)
            
            if len(pop)>0:
                # 2. find the worst & best
                rank = np.argsort(fitness) 
                best_idx = rank[0].item()
                worst_idx = rank[-1].item()

                # ====== record ======
                D[:nqry] = l0(self.modify(pop[best_idx],oimg,timg),oimg)
                # ====================
                
                # 3. evolution
                while True:
                    # a. Crossover (recombine)
                    idxs = [idx for idx in range(self.pop_size) if idx != best_idx]
                    id1, id2 = np.random.choice(idxs, 2, replace = False)
                    offspring = self.recombine(pop[best_idx],pop[id1],pop[id2])

                    # b. mutation (diversify)
                    offspring = self.mutate(offspring)
                        
                    # c. fitness evaluation
                    nqry += 1
                    fo = self.feval(offspring,oimg,timg,olabel,tlabel)
                        
                    # d. select
                    pop[worst_idx],fitness[worst_idx] = self.selection(pop[worst_idx],fitness[worst_idx],offspring,fo)
                        
                    # e. update best and worst
                    rank = np.argsort(fitness)
                    best_idx = rank[0].item()
                    worst_idx = rank[-1].item()

                    # ====== record ======
                    D[nqry] = l0(self.modify(pop[best_idx],oimg,timg),oimg)
                    # ====================
                    
                    if nqry % 200 == 0:
                        print(pop[best_idx].sum().item(),nqry,self.model.predict_label(self.modify(pop[best_idx],oimg,timg)))
                    if nqry > max_query:
                        break
                
                # ====================

                adv = self.modify(pop[best_idx],oimg,timg)
            else:
                adv = timg
                D[:nqry] = len(self.masking(oimg,timg))
        else:
            adv = timg
            nqry = 1 # output purpose, not mean number of qry = 1
            D[0] = 1
            
        return adv, nqry, D[:nqry]