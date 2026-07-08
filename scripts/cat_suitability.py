"""Scan SMILES for cat-move suitability, separating SPECIFIC (fitted-library) shared
categories from GENERIC (sp3_sp3/sp3_sp2) lumping. tied_spec = # dihedrals in a specific
category shared by >=2 dihedrals (true repeat structure cat can exploit)."""
import sys, argparse, logging, collections
logging.disable(logging.CRITICAL)
from rdkit import Chem
from bouquet.setup import detect_dihedrals
from bouquet.priors import create_prior_module
GENERIC={'sp3_sp3','sp3_sp2','uniform'}
def scan_one(smiles, priors='gfn2_priors.json'):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None: return None
    mol=Chem.AddHs(mol)
    dih=detect_dihedrals(mol); d=len(dih)
    if d==0: return (0,0,0,0)
    pm=create_prior_module(mol=mol, dihedrals=[x.chain for x in dih], univariate_file=priors)
    ua=pm.univariate_assignments
    bycat=collections.Counter(ua.get(i) for i in range(d))  # type -> count (None=unassigned)
    tied_spec=sum(c for t,c in bycat.items() if c>=2 and isinstance(t,int))
    tied_gen =sum(c for t,c in bycat.items() if c>=2 and t in GENERIC)
    max_spec =max([c for t,c in bycat.items() if isinstance(t,int)] or [0])
    return (d,tied_spec,tied_gen,max_spec)
def parse(path):
    out=[]
    for ln in open(path):
        ln=ln.strip()
        if not ln: continue
        p=ln.split(); out.append((p[0],p[1] if len(p)>1 else f'm{len(out)}'))
    return out
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('smiles_file')
    ap.add_argument('--priors', default='gfn2_priors.json', help='univariate priors JSON file')
    args=ap.parse_args()
    print('name\td\ttied_spec\ttied_gen\tmax_spec')
    for smi,name in parse(args.smiles_file):
        try:
            r = scan_one(smi, args.priors)
        except Exception as e:
            print(f"warning: skip {name}: {e}", file=sys.stderr)
            continue
        if r is None:
            continue
        print(f'{name}\t{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}')
