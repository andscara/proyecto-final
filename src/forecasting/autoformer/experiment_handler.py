from abc import ABC, abstractmethod
from enum import Enum
import duckdb as ddb
from forecasting.autoformer.data_loader import data_splitter
from dataclasses import dataclass
from forecasting.autoformer.data_loader import WindowsDataset
from forecasting.autoformer.experiment_configuration import ExperimentConfiguration

@dataclass
class ExperimentGroup:
    """Data class for grouping experiments."""
    name: str
    full_dataset: WindowsDataset
    train_dataset: WindowsDataset
    val_dataset: WindowsDataset
    test_dataset: WindowsDataset


class BaseExperimentHandler(ABC):

    def __init__(
        self, 
        db_path: str, 
        data_path: str,
        exp_config: ExperimentConfiguration
    ):
        self._db_path = db_path
        self._data_path = data_path
        self._exp_config = exp_config

    @abstractmethod
    def has_next(self) -> bool:
        """
        Check if there are more experiments to run.
        """
        ...

    @abstractmethod
    def next_experiment_group(self) -> ExperimentGroup:
        """
        Get the next experiment group to run.
        """
        ...

    @abstractmethod
    def use_exogenous(self) -> bool:
        ...

class Region(Enum):
    NORTH = ("NORTH", ["ARTIGAS", "SALTO", "RIVERA", "TACUAREMBO", "CERRO LARGO"])
    SOUTH = ("SOUTH", ["SAN JOSE", "COLONIA", "CANELONES", "FLORES", "FLORIDA", "SORIANO"])
    EAST = ("EAST", ["MALDONADO", "ROCHA", "TREINTA Y TRES", "LAVALLEJA"])
    WEST = ("WEST", ["PAYSANDU","RIO NEGRO", "DURAZNO"])
    MONTEVIDEO = ("MONTEVIDEO", ["MONTEVIDEO"])

    def __init__(self, code: str, departamentos: list[str]):
        self.code = code
        self.departamentos = departamentos

class RegionsExperimentHandler(BaseExperimentHandler):

    def __init__(self, db_path: str, data_path: str, exp_config: ExperimentConfiguration):
        super().__init__(db_path, data_path, exp_config)
        self._region_index = 0

    def has_next(self) -> bool:
        return self._region_index < Region.__len__()
    
    def next_experiment_group(self) -> ExperimentGroup:
        region = list(Region)[self._region_index]
        self._region_index += 1
        query = f"""
        select e.dia, e.hora, SUM(agg_valor) as agg_valor, AVG((temperatura + 15) / 65) as temp_media
        from (
            select departamento, dia, hora, SUM(valor) as agg_valor
            from read_parquet('{self._data_path}')
            where departamento in {tuple(region.departamentos)}
            group by departamento, dia, hora
        ) e inner join temp_departamento t on e.dia=t.dia and e.hora=t.hora and t.departamento = e.departamento
        group by e.dia, e.hora
        order by e.dia, e.hora
        """

        con = ddb.connect(database=self._db_path)
        ts_agg_region = con.execute(query).fetchdf()
        print(f"Cantidad de registros totales en todos los departamentos agregados: {len(ts_agg_region)}")
        con.close()
        print ("Creating datasets...")
        all_dataset, train_dataset, val_dataset, test_dataset = data_splitter(
            df=ts_agg_region,
            exp_config=self._exp_config
        )
        print ("Datasets created.")
        return ExperimentGroup(
            name=f"Region {region.code}",
            full_dataset=all_dataset,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset
        )
    
    def use_exogenous(self) -> bool:
        return True
    
class CountryExperimentHandler(BaseExperimentHandler):

    def __init__(self, db_path: str, data_path: str, exp_config: ExperimentConfiguration):
        super().__init__(db_path, data_path, exp_config)
        self._has_next = True

    def has_next(self) -> bool:
        has_next = self._has_next
        self._has_next = False
        return has_next
    
    def next_experiment_group(self) -> ExperimentGroup:
        query = f"""
        select e.dia, e.hora, SUM(e.valor) as agg_valor, AVG((t.temperatura + 15) / 65) as temp_media
        from read_parquet('{self._data_path}') e inner join temp_departamento t on e.dia=t.dia and e.hora=t.hora and t.departamento = e.departamento
        group by e.dia, e.hora
        order by e.dia, e.hora
        """

        con = ddb.connect(database=self._db_path)
        ts_agg_region = con.execute(query).fetchdf()
        print(f"Cantidad de registros totales en todo el pais: {len(ts_agg_region)}")
        con.close()
        print ("Creating datasets...")
        all_dataset, train_dataset, val_dataset, test_dataset = data_splitter(
            df=ts_agg_region,
            exp_config=self._exp_config
        )
        print ("Datasets created.")
        return ExperimentGroup(
            name=f"Country",
            full_dataset=all_dataset,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset
        )
    
    def use_exogenous(self) -> bool:
        return False

class ExperimentType(Enum):
    REGIONS = "regions"
    COUNTRY = "country"
    DEPARTAMENTS = "departments"
    REGION_CLUSTERING = "region_clustering"


def experiment_factory(
    experiment_type: ExperimentType,
    db_path: str, 
    data_path: str, 
    exp_config: ExperimentConfiguration
) -> BaseExperimentHandler:
    if experiment_type == ExperimentType.REGIONS:
        return RegionsExperimentHandler(db_path, data_path, exp_config)
    elif experiment_type == ExperimentType.COUNTRY:
        return CountryExperimentHandler(db_path, data_path, exp_config)
    else:
        raise ValueError(f"Experiment type {experiment_type} not supported."
)


