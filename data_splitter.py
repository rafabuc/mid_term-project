"""
DATA SPLITTER 
========================

Load sampled_metadata.csv, divide en train/test and save.

Input:
- sampled_metadata.csv

Output:
- train_metadata.csv
- test_metadata.csv
- split_info.json

"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Divide sampled_metadata en train/test y guarda los archivos.
    
    Attributes:
        test_size (float): Proporción de datos para test
        random_state (int): Semilla para reproducibilidad
        target_column (str): Nombre de la columna objetivo
        split_info (Dict): Información del split realizado
    """
    
    def __init__(
        self,
        test_size: float, # = 0.2,
        random_state: int, # = 42,
        target_column: str , #= 'main_category',
        train_file_name: str ,#= 'train_metadata.csv',
        test_file_name: str ,#= 'test_metadata.csv',
        info_file_name: str #= 'split_info.json'
    ):
        """
        Inicializa el DataSplitter.
        
        Args:
            test_size: Proporción de datos para test (0.2 = 20%)
            random_state: Semilla para reproducibilidad
            target_column: Nombre de la columna objetivo
        """
        self.test_size = test_size
        self.random_state = random_state
        self.target_column = target_column
        self.split_info: Dict = {}
        self.train_file_name = train_file_name
        self.test_file_name = test_file_name    
        self.info_file_name = info_file_name
        logger.info("DataSplitter initialized")
        logger.info(f"  test_size: {test_size}")
        logger.info(f"  random_state: {random_state}")
        logger.info(f"  target_column: {target_column}")
    
    def load_data(self, sampled_metadata_path: str) -> pd.DataFrame:
        """
        Carga el archivo sampled_metadata.csv.
        
        Args:
            sampled_metadata_path: Ruta al archivo CSV
            
        Returns:
            DataFrame con los datos cargados
        """
        logger.info("=" * 80)
        logger.info("LOADING SAMPLED METADATA")
        logger.info("=" * 80)
        
        if not Path(sampled_metadata_path).exists():
            raise FileNotFoundError(f"Archivo no encontrado: {sampled_metadata_path}")
        
        logger.info(f"Loading: {sampled_metadata_path}")
        df = pd.read_csv(sampled_metadata_path)
        
        logger.info(f"✓ Data loaded successfully")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Columns: {list(df.columns)}")
        
        # Verificar que existe la columna objetivo
        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Información de la columna objetivo
        logger.info(f"\nTarget column: {self.target_column}")
        logger.info(f"  Unique values: {df[self.target_column].nunique()}")
        logger.info(f"  Missing values: {df[self.target_column].isna().sum()}")
        
        return df
    
    def split_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Divide los datos en train y test con split estratificado.
        
        Args:
            df: DataFrame con los datos
            
        Returns:
            df_train, df_test
        """
        logger.info("\n" + "=" * 80)
        logger.info("TRAIN/TEST SPLIT")
        logger.info("=" * 80)
        
        # Remover filas con target NaN
        n_before = len(df)
        df_clean = df[df[self.target_column].notna()].copy()
        n_after = len(df_clean)
        
        if n_before != n_after:
            n_removed = n_before - n_after
            logger.warning(f"⚠ Removed {n_removed} rows with NaN in target column")
        
        logger.info(f"Total samples: {len(df_clean):,}")
        logger.info(f"Test size: {self.test_size * 100:.1f}%")
        logger.info(f"Train size: {(1 - self.test_size) * 100:.1f}%")
        
        # Split estratificado para mantener distribución de clases
        df_train, df_test = train_test_split(
            df_clean,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df_clean[self.target_column]
        )
        
        logger.info(f"\n✓ Split completed:")
        logger.info(f"  Train: {len(df_train):,} samples ({len(df_train)/len(df_clean)*100:.1f}%)")
        logger.info(f"  Test:  {len(df_test):,} samples ({len(df_test)/len(df_clean)*100:.1f}%)")
        
        # Analizar distribución de clases en cada conjunto
        self._analyze_class_distribution(df_train, df_test)
        
        # Guardar información del split
        self._save_split_info(df_train, df_test)
        
        return df_train, df_test
    
    def _analyze_class_distribution(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame
    ) -> None:
        """
        Analiza y muestra la distribución de clases en train y test.
        """
        logger.info("\n" + "-" * 80)
        logger.info("CLASS DISTRIBUTION ANALYSIS")
        logger.info("-" * 80)
        
        # Train distribution
        train_dist = df_train[self.target_column].value_counts().sort_index()
        logger.info("\nTRAIN distribution:")
        for category, count in train_dist.items():
            pct = (count / len(df_train)) * 100
            logger.info(f"  {category:30s}: {count:5,} ({pct:5.1f}%)")
        
        # Test distribution
        test_dist = df_test[self.target_column].value_counts().sort_index()
        logger.info("\nTEST distribution:")
        for category, count in test_dist.items():
            pct = (count / len(df_test)) * 100
            logger.info(f"  {category:30s}: {count:5,} ({pct:5.1f}%)")
        
        # Verificar que ambos conjuntos tienen todas las clases
        train_classes = set(train_dist.index)
        test_classes = set(test_dist.index)
        
        if train_classes != test_classes:
            missing_in_test = train_classes - test_classes
            missing_in_train = test_classes - train_classes
            
            if missing_in_test:
                logger.warning(f"⚠ Classes missing in TEST: {missing_in_test}")
            if missing_in_train:
                logger.warning(f"⚠ Classes missing in TRAIN: {missing_in_train}")
        else:
            logger.info("\n✓ All classes present in both TRAIN and TEST")
    
    def _save_split_info(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame
    ) -> None:
        """
        Guarda información sobre el split realizado.
        """
        train_dist = df_train[self.target_column].value_counts().to_dict()
        test_dist = df_test[self.target_column].value_counts().to_dict()
        
        self.split_info = {
            'test_size': self.test_size,
            'random_state': self.random_state,
            'target_column': self.target_column,
            'total_samples': len(df_train) + len(df_test),
            'train_samples': len(df_train),
            'test_samples': len(df_test),
            'n_classes': df_train[self.target_column].nunique(),
            'train_distribution': train_dist,
            'test_distribution': test_dist,
            'classes': sorted(df_train[self.target_column].unique().tolist())
        }
    
    def save_splits(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        output_dir: str = 'data/processed'
    ) -> Dict[str, str]:
        """
        Guarda los datasets de train y test en archivos CSV.
        
        Args:
            df_train: DataFrame de entrenamiento
            df_test: DataFrame de test
            output_dir: Directorio donde guardar los archivos
            
        Returns:
            Dictionary con las rutas de los archivos guardados
        """
        logger.info("\n" + "=" * 80)
        logger.info("SAVING TRAIN/TEST SPLITS")
        logger.info("=" * 80)
        
        # Crear directorio si no existe
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # Definir rutas de salida
        train_path = output_path / self.train_file_name
        test_path = output_path / self.test_file_name
        info_path = output_path / self.info_file_name

        # Guardar train
        logger.info(f"\nSaving train data...")
        df_train.to_csv(train_path, index=False)
        logger.info(f"✓ Train saved: {train_path}")
        logger.info(f"  Shape: {df_train.shape}")
        
        # Guardar test
        logger.info(f"\nSaving test data...")
        df_test.to_csv(test_path, index=False)
        logger.info(f"✓ Test saved: {test_path}")
        logger.info(f"  Shape: {df_test.shape}")
        
        # Guardar información del split
        logger.info(f"\nSaving split info...")
        with open(info_path, 'w') as f:
            json.dump(self.split_info, f, indent=2)
        logger.info(f"✓ Split info saved: {info_path}")
        
        logger.info("\n" + "=" * 80)
        logger.info("ALL FILES SAVED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return {
            'train': str(train_path),
            'test': str(test_path),
            'split_info': str(info_path)
        }
    
    def run_pipeline(
        self,
        sampled_metadata_path: str,
        output_dir: str = 'data/processed'
    ) -> Dict:
        """
        Ejecuta el pipeline completo: cargar -> dividir -> guardar.
        
        Args:
            sampled_metadata_path: Ruta a sampled_metadata.csv
            output_dir: Directorio donde guardar los outputs
            
        Returns:
            Dictionary con resultados y rutas de archivos
        """
        logger.info("\n" + "=" * 80)
        logger.info("RUNNING DATA SPLITTER PIPELINE")
        logger.info("=" * 80)
        
        # 1. Cargar datos
        df = self.load_data(sampled_metadata_path)
        
        # 2. Dividir en train/test
        df_train, df_test = self.split_data(df)
        
        # 3. Guardar archivos
        file_paths = self.save_splits(df_train, df_test, output_dir)
        
        # Resumen final
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("\nGenerated files:")
        for key, path in file_paths.items():
            logger.info(f"  {key}: {path}")
        
        logger.info("\nNext steps:")
        logger.info("  → Use train_metadata.csv for feature extraction")
        logger.info("  → Use test_metadata.csv for final evaluation")
        logger.info("  → Check split_info.json for detailed statistics")
        logger.info("=" * 80)
        
        return {
            'df_train': df_train,
            'df_test': df_test,
            'file_paths': file_paths,
            'split_info': self.split_info
        }


