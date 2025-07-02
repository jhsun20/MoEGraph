from models.gnn_models import GCN, GIN, GraphSAGE
from models.experts import MoEModel
from models.UIL import UILModel
from models.UILexperts import MoEUILModel
from models.UILexpertsShared import MoEUILModelSharedEncoder

def get_model(config, dataset_info):
    """
    Factory function to create a model based on configuration.
    
    Args:
        config (dict): Configuration dictionary
        dataset_info (dict): Dataset information including num_features and num_classes
        
    Returns:
        nn.Module: Instantiated model
    """
    model_config = config['model']
    model_type = model_config['type']
    
    # Common parameters
    num_features = dataset_info['num_features']
    num_classes = dataset_info['num_classes']
    hidden_dim = model_config['hidden_dim']
    num_layers = model_config['num_layers']
    dropout = model_config['dropout']
    pooling = model_config['pooling']
    
    # Create model based on type
    if model_type == 'GCN':
        model = GCN(num_features, num_classes, hidden_dim, num_layers, dropout, pooling)
    elif model_type == 'GIN':
        model = GIN(num_features, num_classes, hidden_dim, num_layers, dropout, pooling)
    elif model_type == 'GraphSAGE':
        model = GraphSAGE(num_features, num_classes, hidden_dim, num_layers, dropout, pooling)
    elif model_type == 'moe':
        model = MoEModel(config, dataset_info)
    elif model_type == 'moe_shared':
        model = MoeModelShared(config, dataset_info)
    elif model_type == 'uil':
        model = UILModel(config, dataset_info)
    elif model_type == 'moe_uil':
        #model = MoEUILModel(config, dataset_info)
        model = MoEUILModelSharedEncoder(config, dataset_info)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model 