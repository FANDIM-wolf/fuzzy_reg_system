import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import warnings

# Игнорируем предупреждения
warnings.filterwarnings('ignore', category=UserWarning)

class ExpertSystemBase:
    def __init__(self, data_path, x_columns, y_column,
                 fuzzy_vars_config, fuzzy_rules,
                 model_type='random_forest', model_params=None,
                 test_size=0.2, random_state=42):
        """
        Shared data module for the expert system with model configuration
        """
        # Set default model parameters if not provided
        if model_params is None:
            model_params = {}

        # Load data
        try:
            self.data = pd.read_csv(data_path, sep=',').rename(columns=lambda x: x.strip())
            print("Loaded columns:", self.data.columns)

            # Validate columns
            if not all(col in self.data.columns for col in x_columns):
                raise ValueError("One or more specified input columns are missing in the data")
            if y_column not in self.data.columns:
                raise ValueError(f"Output column '{y_column}' is missing in the data")

            # Store input and output data
            self.X_regression = self.data[x_columns]
            self.y_regression = self.data[y_column]

            # Model selection and parameters
            self.model_type = model_type
            self.model_params = model_params

            # Split train/test data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X_regression, self.y_regression,
                test_size=test_size,
                random_state=random_state
            )

            # Define fuzzy sets
            self.fuzzy_sets = self._define_fuzzy_sets(fuzzy_vars_config)

            # Store rules
            self.fuzzy_rules = [rule for rule in fuzzy_rules if ' or ' not in rule]

            # Input data for fuzzy model
            self.fuzzy_inputs = self.data[x_columns]

        except Exception as e:
            print(f"Data loading error: {e}")
            raise

    def _define_fuzzy_sets(self, fuzzy_vars_config):
        """Create fuzzy sets with specified shape"""
        fuzzy_sets = {}
        for var_name, config in fuzzy_vars_config.items():
            # For auto-generated sets
            if 'min' in config and 'max' in config and 'triangles' in config:
                min_val = config['min']
                max_val = config['max']
                num_triangles = config['triangles']
                shape = config.get('shape', 'triangular')

                # Calculate step size between triangle centers
                if num_triangles > 1:
                    step = (max_val - min_val) / (num_triangles - 1)
                else:
                    step = 0

                sets = {}
                for i in range(num_triangles):
                    # Calculate center of current triangle
                    center = min_val + i * step

                    # Calculate left and right points
                    left = center - step if i > 0 else min_val
                    right = center + step if i < num_triangles - 1 else max_val

                    # Ensure left doesn't go below min, right doesn't exceed max
                    left = max(min_val, left)
                    right = min(max_val, right)

                    sets[f'level_{i}'] = {
                        'shape': shape,
                        'params': [left, center, right] if shape == 'triangular' else [center, step/2]
                    }

                fuzzy_sets[var_name] = sets

            # For manually defined sets
            elif 'terms' in config:
                fuzzy_sets[var_name] = {}
                for term, term_config in config['terms'].items():
                    fuzzy_sets[var_name][term] = {
                        'shape': term_config['shape'],
                        'params': term_config['params']
                    }
            else:
                raise ValueError(f"Invalid configuration for fuzzy variable: {var_name}")

        return fuzzy_sets

    def visualize_fuzzy_sets(self):
        """Visualize membership functions with proper overlap"""
        for var_name, sets in self.fuzzy_sets.items():
            fig, ax = plt.subplots(figsize=(10, 4))

            # Create universe for plotting
            all_points = []
            for term_config in sets.values():
                if term_config['shape'] == 'triangular':
                    all_points.extend(term_config['params'])
                elif term_config['shape'] == 'gaussian':
                    mean, sigma = term_config['params']
                    all_points.extend([mean - 3*sigma, mean + 3*sigma])

            min_val = min(all_points)
            max_val = max(all_points)
            universe = np.linspace(min_val, max_val, 1000)

            # Plot each membership function
            for term, term_config in sets.items():
                shape = term_config['shape']
                params = term_config['params']

                if shape == 'triangular':
                    mf = fuzz.trimf(universe, params)
                elif shape == 'gaussian':
                    mf = fuzz.gaussmf(universe, *params)
                else:
                    continue

                ax.plot(universe, mf, label=f"{term} ({shape})")

            ax.set_title(f'Membership functions: {var_name}')
            ax.legend()
            ax.grid(True)
            ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
            plt.show()

    # Data access methods
    def get_regression_train(self):
        return self.X_train, self.y_train

    def get_regression_test(self):
        return self.X_test, self.y_test

    def get_fuzzy_inputs(self):
        return self.fuzzy_inputs

    def get_fuzzy_sets(self):
        return self.fuzzy_sets

    def get_fuzzy_rules(self):
        return self.fuzzy_rules

    def get_model_type(self):
        return self.model_type

    def get_model_params(self):
        return self.model_params


class RegressionModule:
    def __init__(self, expert_system):
        """Regression module with configurable model type and parameters"""
        self.expert_system = expert_system
        self.model_type = expert_system.get_model_type()
        self.model_params = expert_system.get_model_params()

        # Initialize selected model with parameters
        if self.model_type == 'random_forest':
            # Set default parameters if not provided
            default_params = {
                'n_estimators': 100,
                'max_depth': None,
                'random_state': 42
            }
            # Merge user parameters with defaults
            params = {**default_params, **self.model_params}
            self.model = RandomForestRegressor(**params)

        elif self.model_type == 'neural_network':
            # Set default parameters if not provided
            default_params = {
                'hidden_layer_sizes': (100,),
                'activation': 'relu',
                'solver': 'adam',
                'max_iter': 1000,
                'random_state': 42
            }
            # Merge user parameters with defaults
            params = {**default_params, **self.model_params}
            self.model = MLPRegressor(**params)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self):
        """Train model with progress monitoring"""
        X_train, y_train = self.expert_system.get_regression_train()
        self.model.fit(X_train, y_train)

    def evaluate(self):
        """Evaluate model quality"""
        X_test, y_test = self.expert_system.get_regression_test()
        predictions = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        return mae, mse

    def predict(self, X):
        """Make prediction"""
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        prediction = self.model.predict(X)
        return float(prediction[0])

    def get_model_info(self):
        """Get information about the trained model"""
        info = f"Model type: {self.model_type}\n"

        if self.model_type == 'random_forest':
            info += f"Number of trees: {self.model.n_estimators}\n"
            info += f"Max depth: {self.model.max_depth or 'Unlimited'}\n"
            info += f"Features: {self.model.n_features_in_}\n"

        elif self.model_type == 'neural_network':
            info += f"Hidden layers: {self.model.hidden_layer_sizes}\n"
            info += f"Activation: {self.model.activation}\n"
            info += f"Solver: {self.model.solver}\n"
            info += f"Final loss: {self.model.loss_:.4f}\n"
            info += f"Training iterations: {self.model.n_iter_}\n"

        return info


class FuzzyModule:
    def __init__(self, expert_system):
        """Fuzzy module with shape selection and linguistic output"""
        self.expert_system = expert_system
        self.fuzzy_sets = expert_system.get_fuzzy_sets()
        self.fuzzy_rules = expert_system.get_fuzzy_rules()
        self.default_output = 5.0
        self.system, self.inputs, self.output = self._create_fuzzy_system()

    def _create_fuzzy_system(self):
        """Create fuzzy system with proper input/output variables and rules"""
        # Create input variables
        inputs = {}
        for var_name, sets in self.fuzzy_sets.items():
            if var_name == 'output':
                continue

            # Create universe for this variable
            all_points = []
            for term_config in sets.values():
                shape = term_config['shape']
                params = term_config['params']

                if shape == 'triangular':
                    all_points.extend(params)
                elif shape == 'gaussian':
                    mean, sigma = params
                    all_points.extend([mean - 3*sigma, mean + 3*sigma])

            universe = np.linspace(min(all_points), max(all_points), 1000)
            inputs[var_name] = ctrl.Antecedent(universe, var_name)

            # Add membership functions
            for term, term_config in sets.items():
                shape = term_config['shape']
                params = term_config['params']

                if shape == 'triangular':
                    inputs[var_name][term] = fuzz.trimf(inputs[var_name].universe, params)
                elif shape == 'gaussian':
                    inputs[var_name][term] = fuzz.gaussmf(inputs[var_name].universe, *params)

        # Create output variable
        output_config = self.fuzzy_sets.get('output', None)
        if not output_config:
            raise ValueError("Output variable not defined in fuzzy_sets")

        all_points = []
        for term_config in output_config.values():
            shape = term_config['shape']
            params = term_config['params']

            if shape == 'triangular':
                all_points.extend(params)
            elif shape == 'gaussian':
                mean, sigma = params
                all_points.extend([mean - 3*sigma, mean + 3*sigma])

        universe = np.linspace(min(all_points), max(all_points), 1000)
        output = ctrl.Consequent(universe, 'output')

        for term, term_config in output_config.items():
            shape = term_config['shape']
            params = term_config['params']

            if shape == 'triangular':
                output[term] = fuzz.trimf(output.universe, params)
            elif shape == 'gaussian':
                output[term] = fuzz.gaussmf(output.universe, *params)

        # Create rules
        rules = []
        for rule_str in self.fuzzy_rules:
            try:
                if 'then' not in rule_str:
                    continue

                antecedent, consequent = rule_str.split('then')
                antecedent = antecedent.replace('If', '').strip()
                consequent = consequent.strip()

                # Parse conditions
                conditions = [cond.strip() for cond in antecedent.split('and')]
                fuzzy_antecedent = None

                for condition in conditions:
                    var_name, term = condition.split('is')
                    var_name = var_name.strip()
                    term = term.strip()

                    if fuzzy_antecedent is None:
                        fuzzy_antecedent = inputs[var_name][term]
                    else:
                        fuzzy_antecedent = fuzzy_antecedent & inputs[var_name][term]

                # Create rule
                _, output_term = consequent.split('is')
                output_term = output_term.strip()
                rules.append(ctrl.Rule(fuzzy_antecedent, output[output_term]))

            except Exception as e:
                print(f"Error parsing rule: {rule_str} - {str(e)}")
                continue

        # Create control system
        control_system = ctrl.ControlSystem(rules)
        return ctrl.ControlSystemSimulation(control_system), inputs, output

    def compute(self, inputs):
        """Perform fuzzy inference and return linguistic term"""
        # Set input values
        for var_name, value in inputs.items():
            if var_name in self.inputs:
                self.system.input[var_name] = value

        try:
            # Compute the result
            self.system.compute()
            crisp_output = self.system.output['output']
        except Exception as e:
            print(f"Fuzzy computation error: {e}")
            crisp_output = self.default_output

        # Determine linguistic term with highest membership
        max_membership = -1
        linguistic_term = "unknown"

        for term in self.output.terms:
            membership = fuzz.interp_membership(
                self.output.universe,
                self.output[term].mf,
                crisp_output
            )

            if membership > max_membership:
                max_membership = membership
                linguistic_term = term

        return crisp_output, linguistic_term


if __name__ == "__main__":
    # Configuration
    data_path = "data.csv"
    x_columns = ['T1', 'T2', 'H1']
    y_column = 'H2'

    # Fuzzy variables configuration with shape selection
    fuzzy_vars_config = {
        'T1': {
            'min': 0,
            'max': 100,
            'triangles': 3,
            'shape': 'gaussian'  # 'triangular' or 'gaussian'
        },
        'T2': {
            'min': 0,
            'max': 100,
            'triangles': 3,
            'shape': 'gaussian'
        },
        'H1': {
            'min': 0,
            'max': 100,
            'triangles': 5,
            'shape': 'triangular'
        },
        'H2': {
            'min': 0,
            'max': 100,
            'triangles': 5,
            'shape': 'triangular'
        },
        'output': {
            'terms': {
                'low': {
                    'shape': 'triangular',
                    'params': [0, 0, 5]
                },
                'medium': {
                    'shape': 'gaussian',
                    'params': [5, 1.5]  # mean, sigma
                },
                'high': {
                    'shape': 'triangular',
                    'params': [5, 10, 10]
                }
            }
        }
    }

    # Fuzzy rules with linguistic terms
    fuzzy_rules = [
        "If T1 is level_0 and T2 is level_0 and H1 is level_0 and H2 is level_0 then output is low",
        "If T1 is level_0 and T2 is level_1 and H1 is level_1 and H2 is level_1 then output is low",
        "If T1 is level_1 and T2 is level_1 and H1 is level_2 and H2 is level_2 then output is medium",
        "If T1 is level_1 and T2 is level_2 and H1 is level_3 and H2 is level_3 then output is medium",
        "If T1 is level_2 and T2 is level_2 and H1 is level_4 and H2 is level_4 then output is high",
        "If T1 is level_0 and T2 is level_2 and H1 is level_0 and H2 is level_4 then output is low",
        "If T1 is level_1 and T2 is level_1 and H1 is level_2 and H2 is level_3 then output is medium",
        "If T1 is level_0 and T2 is level_1 and H1 is level_0 and H2 is level_1 then output is low",
        "If T1 is level_2 and T2 is level_1 and H1 is level_4 and H2 is level_3 then output is medium",
        "If T1 is level_1 and T2 is level_2 and H1 is level_3 and H2 is level_4 then output is high",
        "If T1 is level_1 and T2 is level_1 and H1 is level_2 and H2 is level_2 then output is medium"
    ]

    # ========= USER CONFIGURATION =========
    # Select model type and configure parameters
    MODEL_TYPE = 'random_forest'  # 'random_forest' or 'neural_network'

    # Configuration for Random Forest
    rf_params = {
        'n_estimators': 150,      # Number of trees
        'max_depth': 8,           # Maximum tree depth
    }

    # Configuration for Neural Network
    nn_params = {
        'hidden_layer_sizes': (64, 32),  # Network architecture
        'activation': 'tanh',            # Activation function
        'solver': 'adam',                # Weight optimization solver
        'max_iter': 1000,                # Maximum number of iterations
    }
    # ======================================

    # Select parameters based on model type
    model_params = rf_params if MODEL_TYPE == 'random_forest' else nn_params

    # Initialize expert system
    print("Initializing expert system...")
    expert_system = ExpertSystemBase(
        data_path=data_path,
        x_columns=x_columns,
        y_column=y_column,
        fuzzy_vars_config=fuzzy_vars_config,
        fuzzy_rules=fuzzy_rules,
        model_type=MODEL_TYPE,
        model_params=model_params
    )

    # Visualize fuzzy sets
    expert_system.visualize_fuzzy_sets()

    # Regression module
    print("\nTraining regression model...")
    regression_module = RegressionModule(expert_system)
    regression_module.train()

    # Display model info
    print("\nModel Information:")
    print(regression_module.get_model_info())

    # Evaluate model
    mae, mse = regression_module.evaluate()
    print(f"Regression model metrics - MAE: {mae:.2f}, MSE: {mse:.2f}")

    # Make prediction
    example_input = {'T1': 25, 'T2': 27, 'H1': 60}
    predicted_H2 = regression_module.predict(example_input)
    print(f"\nPredicted H2 value: {predicted_H2:.2f}")

    # Fuzzy inference
    print("\nPerforming fuzzy inference...")
    fuzzy_inputs = {**example_input, 'H2': predicted_H2}
    print("Fuzzy inputs:", fuzzy_inputs)

    fuzzy_module = FuzzyModule(expert_system)
    crisp_output, linguistic_term = fuzzy_module.compute(fuzzy_inputs)
    print(f"Fuzzy output: {crisp_output:.2f} ({linguistic_term})")