#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:37:35 2019

@author: ryb003
"""

from SPARQLWrapper import SPARQLWrapper, JSON
from string import Template
import pandas as pd
import pickle
import os

# path = os.path.dirname(__file__)+'/'
path = "./query_expansion"


def runQuery(q_template, param_value_dict, return_vars, params={}, endpoint="http://sparql.bioontology.org/sparql"):
    t = Template(q_template)
    q = t.substitute(**param_value_dict)
    sparql = SPARQLWrapper(endpoint)
    for key in params:
        sparql.addCustomParameter(key, params[key])
    sparql.setQuery(q)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    output=[[] for v in return_vars]
    for result in results["results"]["bindings"]:
        i=0
        for v in return_vars:
            output[i].append(result[v]["value"])
            i+=1
    return output


def findDiseaseURIByName(d, endpoint="http://sparql.bioontology.org/sparql", params={}):
    templ="""
        SELECT DISTINCT ?x ?name
        FROM <http://bioportal.bioontology.org/ontologies/SNOMEDCT>
        WHERE
        {
            ?x ?p "$disease"@En.
            ?x <http://www.w3.org/2004/02/skos/core#prefLabel> ?name
        }

        """

    res = runQuery(
        templ,
        {'disease': d},
        ['x', 'name'],
        params=params,
        endpoint=endpoint
    )
    return res[0]

def findPredValues(uri, endpoint="http://sparql.bioontology.org/sparql", params={},
              predicate='http://www.w3.org/2004/02/skos/core#prefLabel'):
    templ="""
        SELECT DISTINCT ?name
        FROM <http://bioportal.bioontology.org/ontologies/SNOMEDCT>
        WHERE
        {
          <$uri> <$pred> ?name
        }
    """
    res = runQuery(
        templ,
        {'uri': uri, 'pred': predicate},
        ['name'],
        params=params,
        endpoint=endpoint
    )
    return res[0]

def findPredSubj(uri, endpoint="http://sparql.bioontology.org/sparql", params={},
              predicate='http://www.w3.org/2004/02/skos/core#prefLabel'):
    templ="""
        SELECT DISTINCT ?u
        FROM <http://bioportal.bioontology.org/ontologies/SNOMEDCT>
        WHERE
        {
          ?u <$pred> <$uri>
        }
    """
    res = runQuery(templ, {'uri':uri, 'pred':predicate}, ['u'],
                 params=params,
                 endpoint=endpoint)
    return res[0]

def expandDisease(disease_name, endpoint="http://sparql.bioontology.org/sparql", params={}):
    uris=findDiseaseURIByName(disease_name, params=params)
    official_names={}
    synonyms={}
    hyponyms={}
    hypernyms={}
    for u in uris:
        official_names[u]=findPredValues(u, params=params)
        synonyms[u]=findPredValues(u, params=params, predicate='http://www.w3.org/2004/02/skos/core#altLabel')

        hypernymUris=findPredValues(u, params=params, predicate='http://www.w3.org/2000/01/rdf-schema#subClassOf')
        hyponymUris=findPredSubj(u, params=params, predicate='http://www.w3.org/2000/01/rdf-schema#subClassOf')

        hyponyms[u]={}
        for h in hyponymUris:
            hyponyms[u][h]={}
            hName=findPredValues(h, params=params)
            hSynonym=findPredValues(h, params=params, predicate='http://www.w3.org/2004/02/skos/core#altLabel')
            hyponyms[u][h]['pref']=hName
            hyponyms[u][h]['alt']=hSynonym

        hypernyms[u]={}
        for h in hypernymUris:
            hypernyms[u][h]={}
            hName=findPredValues(h, params=params)
            hSynonym=findPredValues(h, params=params, predicate='http://www.w3.org/2004/02/skos/core#altLabel')
            hypernyms[u][h]['pref']=hName
            hypernyms[u][h]['alt']=hSynonym

    # discard ambigous concepts if they're not the only ones
    output=(official_names, synonyms, hyponyms, hypernyms)
    ambig=[]
    for u in official_names:
        if 'http://purl.bioontology.org/ontology/SNOMEDCT/363660007' in hypernyms[u]:
            hypernyms[u].pop('http://purl.bioontology.org/ontology/SNOMEDCT/363660007')
            ambig.append(u)
    if len(ambig)<len(official_names):
        for qe_dict in output:
            for u in ambig:
                qe_dict.pop(u)
    return official_names, synonyms, hyponyms, hypernyms


def expandGene(gene_info, file=path+'Homo_sapiens.gene_info'):
    df = pd.read_csv(f'{path}/Homo_sapiens.gene_info', sep='\t')
    tokens = gene_info.translate(str.maketrans('(),.;', '     ')).split()
    tokens = [t for t in tokens if t.isupper()]
    output = {}
    for t in tokens:
        x = df.loc[df['Symbol'] == t]
        if len (x) > 0:
            x['Synonyms'].values[0].split('|')
            synonyms=x['Synonyms'].values[0].split('|')
            descr=x['description'].values[0]
            full_name=x['Full_name_from_nomenclature_authority'].values[0]
            other=x['Other_designations'].values[0].split('|')
            output[t]=synonyms, descr, full_name, other
    return output


def run(topic_elements, topic, **kwargs):
    label_suffix = '_kbqe' if 'label_suffix' not in kwargs else kwargs['label_suffix']
    sparql_params={} if 'sparql_params' not in kwargs else kwargs['sparql_params']
    endpoint="http://sparql.bioontology.org/sparql" if 'endpoint' not in kwargs else kwargs['endpoint']

    diseases={}
    cache = True if 'cache' not in kwargs else kwargs['cache']
    if cache and os.path.exists(path+'cache.bin'):
        diseases = pickle.load( open( path+'cache.bin', "rb" ) )
    cache_change=False

    for element in topic_elements:
        if element in topic:
            if element == 'disease':
                if topic[element].strip().capitalize() in diseases:
                    disease_qe=diseases[topic[element].strip().capitalize()]
                else:
                    disease_qe=expandDisease(topic[element].strip().capitalize(),
                                         params=sparql_params, endpoint=endpoint)
                    diseases[topic[element].strip().capitalize()]=disease_qe
                    cache_change=True
                preferred_name=''
                for u in disease_qe[0]:
                    for val in disease_qe[0][u]:
                        preferred_name+=val + ' '
                preferred_name=preferred_name.strip()
                synonyms=''
                for u in disease_qe[1]:
                    for val in disease_qe[1][u]:
                        synonyms+=val + ' '
                synonyms=synonyms.strip()
                hyponyms=''
                for u in disease_qe[2]:
                    for val in disease_qe[2][u]:
                        hyponyms+=' '.join(disease_qe[2][u][val]['pref']) + ' '
                hyponyms=hyponyms.strip()
                hypernyms=''
                for u in disease_qe[3]:
                    for val in disease_qe[3][u]:
                        hypernyms+=' '.join(disease_qe[3][u][val]['pref']) + ' '
                hypernyms=hypernyms.strip()

                topic[element+label_suffix+'_pn']=preferred_name
                topic[element+label_suffix+'_syn']=synonyms
                topic[element+label_suffix+'_hypo']=hyponyms
                topic[element+label_suffix+'_hyper']=hypernyms

            if element=='gene':
                gene_qe=expandGene(topic[element])
                synonyms=''
                official_names=''
                other=''
                for gene in gene_qe:
                    for s in gene_qe[gene][0]:
                        synonyms+=s+ ' '
                    official_names+=gene_qe[gene][2] + ' '
                    for o in gene_qe[gene][3]:
                        other+= o + ' '
                synonyms=synonyms.strip()
                official_names=official_names.strip()
                other=other.strip()

                topic[element+label_suffix+'_fn']=official_names
                topic[element+label_suffix+'_syn']=synonyms
                topic[element+label_suffix+'_other']=other

    if cache and cache_change:
        # path = "./"
        print(f"path is: {path}")
        pickle.dump(diseases, open(f"{path}/disease.pickle", "wb"))
    return topic

if __name__ == "__main__":

    params = {'apikey':'8de61230-b5f0-42f4-8b3e-8f0b0f426cc1'}
    # params = {}
    t = run(['gene', 'disease'],{'disease':'melanoma', 'gene':'NF2 (K322), AKT1(E17K)'}, **{'sparql_params': params})
    print (t)
    for key in t.keys():
        print(key)
        print(t[key])
        print("----")
