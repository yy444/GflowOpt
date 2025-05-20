import pandas as pd
import urllib.request
import gzip

import pandas as pd
from pgmpy.models import BayesianNetwork

from pathlib import Path
from numpy.random import default_rng
from pgmpy.utils import get_example_model

from dag_gflownet.utils.graph import sample_erdos_renyi_linear_gaussian
from dag_gflownet.utils.sampling import sample_from_linear_gaussian


def download(url, filename):
    if filename.is_file():
        return filename
    filename.parent.mkdir(exist_ok=True)

    # Download & uncompress archive
    with urllib.request.urlopen(url) as response:
        with gzip.GzipFile(fileobj=response) as uncompressed:
            file_content = uncompressed.read()

    with open(filename, 'wb') as f:
        f.write(file_content)
    
    return filename


def get_data(name, args, rng=default_rng()):
    if name == 'erdos_renyi_lingauss':
        graph = sample_erdos_renyi_linear_gaussian(
            num_variables=args.num_variables,
            num_edges=args.num_edges,
            loc_edges=0.0,
            scale_edges=1.0,
            obs_noise=0.1,
            rng=rng
        )
        data = sample_from_linear_gaussian(
            graph,
            num_samples=args.num_samples,
            rng=rng
        )
        score = 'bge'

    elif name == 'sachs_continuous':
        graph = get_example_model('sachs')
        filename = download(
            'https://www.bnlearn.com/book-crc/code/sachs.data.txt.gz',
            Path('data/sachs.data.txt')
        )
        data = pd.read_csv(filename, delimiter='\t', dtype=float)
        data = (data - data.mean()) / data.std()  # Standardize data
        score = 'bge'

    elif name =='sachs_interventional':
        graph = get_example_model('sachs')
        filename = download(
            'https://www.bnlearn.com/book-crc/code/sachs.interventional.txt.gz',
            Path('data/sachs.interventional.txt')
        )
        data = pd.read_csv(filename, delimiter=' ', dtype='category')
        score = 'bde'

    elif name =='asia_interventional':
        graph = get_example_model('asia')
        filename = Path('G:\code\data\\asia.interventional.txt')
        
        data = pd.read_csv(filename, delimiter=' ', dtype='category')
        score = 'bde'
    


    elif name =='asia_interventional_bic':
        
        graph = get_example_model('asia')
        samples = graph.simulate(int(5000), seed=2)

        # fixed the order of the variables
        fixed_order = ['asia', 'tub', 'smoke', 'lung', 'bronc', 'either', 'xray', 'dysp']
        data = samples[fixed_order]

        score = 'bic'
    
    elif name =='alarm_interventional_bic':
        
        graph = get_example_model('alarm')
        samples = graph.simulate(int(5000), seed=2)

        # fixed the order of the variables
        #fixed_order = ['asia', 'tub', 'smoke', 'lung', 'bronc', 'either', 'xray', 'dysp']
        fixed_order = ['HISTORY', 'HREKG', 'LVFAILURE', 'ERRLOWOUTPUT', 'HRSAT', 'VENTALV', 'FIO2', 'VENTLUNG', 'STROKEVOLUME', 'LVEDVOLUME', 'BP', 'CO', 'HYPOVOLEMIA', 'INTUBATION', 'TPR', 'VENTMACH', 'CATECHOL', 'PULMEMBOLUS', 'MINVOL', 'CVP', 'INSUFFANESTH', 'HRBP', 'SAO2', 'HR', 'PRESS', 'ERRCAUTER', 'PVSAT', 'VENTTUBE', 'KINKEDTUBE', 'DISCONNECT', 'MINVOLSET', 'ANAPHYLAXIS', 'EXPCO2', 'ARTCO2', 'PCWP', 'SHUNT', 'PAP']
        data = samples[fixed_order]

        score = 'bic'


    elif name =='sachs_interventional_bic':
        
        graph = get_example_model('sachs')
        samples = graph.simulate(int(50000), seed=2)

        fixed_order = ['Akt', 'Erk', 'Jnk', 'Mek' ,'P38' ,'PIP2' ,'PIP3', 'PKA', 'PKC' ,'Plcg', 'Raf']

        data = samples[fixed_order]

        score = 'bic'
    
    elif name =='child_interventional_bic':
        
        graph = get_example_model('child')
        samples = graph.simulate(int(5000), seed=9)

        fixed_order = ['BirthAsphyxia', 'HypDistrib', 'HypoxiaInO2', 'CO2', 'ChestXray' ,'Grunting', 'LVHreport', 'LowerBodyO2' ,'RUQO2', 'CO2Report' ,'XrayReport' ,'Disease', 'GruntingReport' ,'Age' ,'LVH', 'DuctFlow', 'CardiacMixing' ,'LungParench', 'LungFlow' ,'Sick']

        
        data = samples[fixed_order]

        score = 'bic'

    elif name =='win95pts_interventional_bic':
        
        graph = get_example_model('win95pts')
        samples = graph.simulate(int(5000), seed=9)

        fixed_order = [
            "AppOK", "DataFile", "AppData", "DskLocal", "PrtSpool", "PrtOn", "PrtPaper", 
            "NetPrint", "PrtDriver", "PrtThread", "EMFOK", "GDIIN", "DrvSet", "DrvOK", 
            "GDIOUT", "PrtSel", "PrtDataOut", "PrtPath", "NtwrkCnfg", "PTROFFLINE", "NetOK", 
            "PrtCbl", "PrtPort", "CblPrtHrdwrOK", "LclOK", "DSApplctn", "PrtMpTPth", "DS_NTOK", 
            "DS_LCLOK", "PC2PRT", "PrtMem", "PrtTimeOut", "FllCrrptdBffr", "TnrSpply", "PrtData", 
            "Problem1", "AppDtGnTm", "PrntPrcssTm", "DeskPrntSpd", "PgOrnttnOK", "PrntngArOK", 
            "ScrnFntNtPrntrFnt", "CmpltPgPrntd", "GrphcsRltdDrvrSttngs", "EPSGrphc", "NnPSGrphc", 
            "PrtPScript", "PSGRAPHIC", "Problem4", "TrTypFnts", "FntInstlltn", "PrntrAccptsTrtyp", 
            "TTOK", "NnTTOK", "Problem5", "LclGrbld", "NtGrbld", "GrbldOtpt", "HrglssDrtnAftrPrnt", 
            "REPEAT", "AvlblVrtlMmry", "PSERRMEM", "TstpsTxt", "GrbldPS", "IncmpltPS", "PrtFile", 
            "PrtIcon", "Problem6", "Problem3", "PrtQueue", "NtSpd", "Problem2", "PrtStatPaper", 
            "PrtStatToner", "PrtStatMem", "PrtStatOff"
        ]

        
        data = samples[fixed_order]

        score = 'bic'

    elif name =='hailfinder_interventional_bic':
        
        graph = get_example_model('hailfinder')
        samples = graph.simulate(int(5000), seed=9)

        fixed_order = [
        'N0_7muVerMo', 'SubjVertMo', 'QGVertMotion', 'CombVerMo', 'AreaMeso_ALS',
        'SatContMoist', 'RaoContMoist', 'CombMoisture', 'AreaMoDryAir', 'VISCloudCov',
        'IRCloudCover', 'CombClouds', 'CldShadeOth', 'AMInstabMt', 'InsInMt',
        'WndHodograph', 'OutflowFrMt', 'MorningBound', 'Boundaries', 'CldShadeConv',
        'CompPlFcst', 'CapChange', 'LoLevMoistAd', 'InsChange', 'MountainFcst',
        'Date', 'Scenario', 'ScenRelAMCIN', 'MorningCIN', 'AMCINInScen',
        'CapInScen', 'ScenRelAMIns', 'LIfr12ZDENSd', 'AMDewptCalPl', 'AMInsWliScen',
        'InsSclInScen', 'ScenRel3_4', 'LatestCIN', 'LLIW', 'CurPropConv',
        'ScnRelPlFcst', 'PlainsFcst', 'N34StarFcst', 'R5Fcst', 'Dewpoints',
        'LowLLapse', 'MeanRH', 'MidLLapse', 'MvmtFeatures', 'RHRatio',
        'SfcWndShfDis', 'SynForcng', 'TempDis', 'WindAloft', 'WindFieldMt',
        'WindFieldPln'
        ]

        
        data = samples[fixed_order]

        score = 'bic'
    elif name == 'property_custom':
        # 定义文件路径 (请根据您的实际路径修改)
        dag_file = r'GflowOpt/data/property/DAGtrue_PROPERTY.csv'
        data_file = r'GflowOpt/data/property/PROPERTY_DATA_part_10.csv'

        # 1. 加载DAG结构
        try:
            dag_df = pd.read_csv(dag_file)
            edges_df = dag_df[dag_df['Dependency'] == '->']
            edges = list(zip(edges_df['Variable 1'], edges_df['Variable 2']))
            
            # 从边中获取所有节点，并确保顺序固定 (按首次出现排序)
            nodes_ordered = list(pd.unique(edges_df[['Variable 1', 'Variable 2']].values.ravel('K')))
            # *** 添加 fixed_order ***
            fixed_order = nodes_ordered 
            print(fixed_order)
            graph = BayesianNetwork(ebunch=edges, latents=set())
            graph.add_nodes_from(fixed_order) # 使用固定顺序的节点列表
            
            print(f"从 {dag_file} 加载DAG结构成功，包含 {len(graph.nodes())} 个节点和 {len(graph.edges())} 条边。")
            print(f"固定节点顺序: {fixed_order}")

        except FileNotFoundError:
            raise FileNotFoundError(f"DAG文件未找到: {dag_file}")
        except Exception as e:
            raise RuntimeError(f"加载DAG文件时出错: {e}")

        # 2. 加载数据
        try:
            data = pd.read_csv(data_file)
            # 验证数据列
            missing_cols = set(fixed_order) - set(data.columns)
            extra_cols = set(data.columns) - set(fixed_order)
            if missing_cols:
                print(f"警告: 数据文件中缺少DAG中的节点: {missing_cols}")
            if extra_cols:
                print(f"警告: 数据文件包含DAG中没有的列: {extra_cols}")
            
            # *** 确保数据列顺序与 fixed_order 一致 ***
            try:
                data = data[fixed_order] 
            except KeyError as e:
                 raise KeyError(f"数据文件缺少必要的列: {e}. 请确保数据文件包含所有在DAG中定义的节点。")

            print(f"从 {data_file} 加载数据成功，包含 {len(data)} 条样本和 {len(data.columns)} 个变量。")

        except FileNotFoundError:
            raise FileNotFoundError(f"数据文件未找到: {data_file}")
        except Exception as e:
            raise RuntimeError(f"加载数据文件时出错: {e}")

        # 3. 设置评分方法
        score = 'bic' 
    

    elif name == 'formed_custom':
        # 定义文件路径 (请根据您的实际路径修改)
        dag_file = r'GflowOpt/data/formed/DAGtrue_FORMED.csv'
        data_file = r'GflowOpt/data/formed/FORMED_DATA_part_10.csv'

        # 1. 加载DAG结构
        try:
            dag_df = pd.read_csv(dag_file)
            edges_df = dag_df[dag_df['Dependency'] == '->']
            edges = list(zip(edges_df['Variable 1'], edges_df['Variable 2']))
            
            # 从边中获取所有节点，并确保顺序固定 (按首次出现排序)
            nodes_ordered = list(pd.unique(edges_df[['Variable 1', 'Variable 2']].values.ravel('K')))
            # *** 添加 fixed_order ***
            fixed_order = nodes_ordered
            graph = BayesianNetwork(ebunch=edges, latents=set())
            graph.add_nodes_from(fixed_order) # 使用固定顺序的节点列表
            
            print(f"从 {dag_file} 加载DAG结构成功，包含 {len(graph.nodes())} 个节点和 {len(graph.edges())} 条边。")
            print(f"固定节点顺序: {fixed_order}")

        except FileNotFoundError:
            raise FileNotFoundError(f"DAG文件未找到: {dag_file}")
        except Exception as e:
            raise RuntimeError(f"加载DAG文件时出错: {e}")

        # 2. 加载数据
        try:
            data = pd.read_csv(data_file)
            # 验证数据列
            missing_cols = set(fixed_order) - set(data.columns)
            extra_cols = set(data.columns) - set(fixed_order)
            if missing_cols:
                print(f"警告: 数据文件中缺少DAG中的节点: {missing_cols}")
            if extra_cols:
                print(f"警告: 数据文件包含DAG中没有的列: {extra_cols}")
            
            # *** 确保数据列顺序与 fixed_order 一致 ***
            try:
                data = data[fixed_order]
            except KeyError as e:
                 raise KeyError(f"数据文件缺少必要的列: {e}. 请确保数据文件包含所有在DAG中定义的节点。")

            print(f"从 {data_file} 加载数据成功，包含 {len(data)} 条样本和 {len(data.columns)} 个变量。")

        except FileNotFoundError:
            raise FileNotFoundError(f"数据文件未找到: {data_file}")
        except Exception as e:
            raise RuntimeError(f"加载数据文件时出错: {e}")

        # 3. 设置评分方法
        score = 'bic' 
    elif name == 'sports_custom':
        # 定义文件路径 (请根据您的实际路径修改)
        dag_file = r'GflowOpt/data/sports/DAGtrue_SPORTS.csv'
        data_file = r'GflowOpt/data/sports/SPORTS_DATA_part_10.csv'

        # 1. 加载DAG结构
        try:
            dag_df = pd.read_csv(dag_file)
            edges_df = dag_df[dag_df['Dependency'] == '->']
            edges = list(zip(edges_df['Variable 1'], edges_df['Variable 2']))
            
            # 从边中获取所有节点，并确保顺序固定 (按首次出现排序)
            nodes_ordered = list(pd.unique(edges_df[['Variable 1', 'Variable 2']].values.ravel('K')))
            # *** 添加 fixed_order ***
            fixed_order = nodes_ordered

            graph = BayesianNetwork(ebunch=edges, latents=set())
            graph.add_nodes_from(fixed_order) # 使用固定顺序的节点列表
            
            print(f"从 {dag_file} 加载DAG结构成功，包含 {len(graph.nodes())} 个节点和 {len(graph.edges())} 条边。")
            print(f"固定节点顺序: {fixed_order}")

        except FileNotFoundError:
            raise FileNotFoundError(f"DAG文件未找到: {dag_file}")
        except Exception as e:
            raise RuntimeError(f"加载DAG文件时出错: {e}")

        # 2. 加载数据
        try:
            data = pd.read_csv(data_file)
            # 验证数据列
            missing_cols = set(fixed_order) - set(data.columns)
            extra_cols = set(data.columns) - set(fixed_order)
            if missing_cols:
                print(f"警告: 数据文件中缺少DAG中的节点: {missing_cols}")
            if extra_cols:
                print(f"警告: 数据文件包含DAG中没有的列: {extra_cols}")
            
            # *** 确保数据列顺序与 fixed_order 一致 ***
            try:
                data = data[fixed_order]
            except KeyError as e:
                 raise KeyError(f"数据文件缺少必要的列: {e}. 请确保数据文件包含所有在DAG中定义的节点。")

            print(f"从 {data_file} 加载数据成功，包含 {len(data)} 条样本和 {len(data.columns)} 个变量。")

        except FileNotFoundError:
            raise FileNotFoundError(f"数据文件未找到: {data_file}")
        except Exception as e:
            raise RuntimeError(f"加载数据文件时出错: {e}")

        # 3. 设置评分方法
        score = 'bic' 
    else:
        raise ValueError(f'Unknown graph type: {name}')

    return graph, data, score
