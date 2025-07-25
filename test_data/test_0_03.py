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
            self.fuzzy_rules = fuzzy_rules

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
            if 'min' in config and 'max' in config and 'terms' in config:
                min_val = config['min']
                max_val = config['max']
                num_terms = config['terms']
                shape = config.get('shape', 'triangular')

                # Calculate step size between term centers
                if num_terms > 1:
                    step = (max_val - min_val) / (num_terms - 1)
                else:
                    step = 0

                sets = {}
                for i in range(num_terms):
                    # Calculate center of current term
                    center = min_val + i * step

                    # Calculate left and right points
                    left = center - step if i > 0 else min_val
                    right = center + step if i < num_terms - 1 else max_val

                    # Ensure left doesn't go below min, right doesn't exceed max
                    left = max(min_val, left)
                    right = min(max_val, right)

                    term_name = f'level_{i}'
                    sets[term_name] = {
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

            if not all_points:
                continue

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
        """Fuzzy module with multiple outputs and linguistic results"""
        self.expert_system = expert_system
        self.fuzzy_sets = expert_system.get_fuzzy_sets()
        self.fuzzy_rules = expert_system.get_fuzzy_rules()
        self.debug_info = []
        self.system, self.inputs, self.outputs = self._create_fuzzy_system()

    def _create_fuzzy_system(self):
        """Create fuzzy system with multiple outputs"""
        # Create input variables
        inputs = {}
        outputs = {}

        # Identify output variables (they have special handling)
        all_vars = list(self.fuzzy_sets.keys())
        input_vars = [var for var in all_vars if not var.startswith('output_')]
        output_vars = [var for var in all_vars if var.startswith('output_')]

        # Create input variables
        for var_name in input_vars:
            sets = self.fuzzy_sets[var_name]

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

            if not all_points:
                continue

            min_val = min(all_points)
            max_val = max(all_points)
            universe = np.linspace(min_val, max_val, 1000)
            inputs[var_name] = ctrl.Antecedent(universe, var_name)

            # Add membership functions
            for term, term_config in sets.items():
                shape = term_config['shape']
                params = term_config['params']

                if shape == 'triangular':
                    inputs[var_name][term] = fuzz.trimf(inputs[var_name].universe, params)
                elif shape == 'gaussian':
                    inputs[var_name][term] = fuzz.gaussmf(inputs[var_name].universe, *params)

        # Create output variables
        for var_name in output_vars:
            sets = self.fuzzy_sets[var_name]

            all_points = []
            for term_config in sets.values():
                shape = term_config['shape']
                params = term_config['params']

                if shape == 'triangular':
                    all_points.extend(params)
                elif shape == 'gaussian':
                    mean, sigma = params
                    all_points.extend([mean - 3*sigma, mean + 3*sigma])

            if not all_points:
                continue

            min_val = min(all_points)
            max_val = max(all_points)
            universe = np.linspace(min_val, max_val, 1000)
            outputs[var_name] = ctrl.Consequent(universe, var_name)

            for term, term_config in sets.items():
                shape = term_config['shape']
                params = term_config['params']

                if shape == 'triangular':
                    outputs[var_name][term] = fuzz.trimf(outputs[var_name].universe, params)
                elif shape == 'gaussian':
                    outputs[var_name][term] = fuzz.gaussmf(outputs[var_name].universe, *params)

        # Create rules
        rules = []
        for rule_idx, rule_str in enumerate(self.fuzzy_rules):
            try:
                self.debug_info.append(f"Rule {rule_idx}: Processing: '{rule_str}'")

                if 'then' not in rule_str:
                    self.debug_info.append(f"Rule {rule_idx}: Missing 'then' keyword - skipped")
                    continue

                # Split rule into condition and conclusion
                parts = rule_str.split('then', 1)
                if len(parts) < 2:
                    self.debug_info.append(f"Rule {rule_idx}: Invalid format - skipped")
                    continue

                antecedent = parts[0].replace('If', '').strip()
                consequent = parts[1].strip()

                # Parse conditions
                conditions = [cond.strip() for cond in antecedent.split('and')]
                fuzzy_antecedent = None

                for condition in conditions:
                    # Improved parsing that handles various spacings
                    if ' is ' not in condition:
                        self.debug_info.append(f"Rule {rule_idx}: Invalid condition format (missing 'is'): '{condition}'")
                        continue

                    # Split only on the first occurrence of ' is '
                    condition_parts = condition.split(' is ', 1)
                    if len(condition_parts) < 2:
                        self.debug_info.append(f"Rule {rule_idx}: Invalid condition '{condition}' - skipped")
                        continue

                    var_name = condition_parts[0].strip()
                    term = condition_parts[1].strip()

                    if var_name not in inputs:
                        self.debug_info.append(f"Rule {rule_idx}: Input variable '{var_name}' not found. Available: {list(inputs.keys())}")
                        continue

                    if term not in inputs[var_name].terms:
                        self.debug_info.append(f"Rule {rule_idx}: Term '{term}' not found for '{var_name}'. Available: {list(inputs[var_name].terms)}")
                        continue

                    if fuzzy_antecedent is None:
                        fuzzy_antecedent = inputs[var_name][term]
                    else:
                        fuzzy_antecedent = fuzzy_antecedent & inputs[var_name][term]

                # Parse consequences (multiple outputs)
                consequences = [cons.strip() for cons in consequent.split('and')]
                fuzzy_consequent = []

                for consequence in consequences:
                    # Improved parsing that handles various spacings
                    if ' is ' not in consequence:
                        self.debug_info.append(f"Rule {rule_idx}: Invalid consequence format (missing 'is'): '{consequence}'")
                        continue

                    # Split only on the first occurrence of ' is '
                    consequence_parts = consequence.split(' is ', 1)
                    if len(consequence_parts) < 2:
                        self.debug_info.append(f"Rule {rule_idx}: Invalid consequence '{consequence}' - skipped")
                        continue

                    output_var = consequence_parts[0].strip()
                    term = consequence_parts[1].strip()

                    if output_var not in outputs:
                        self.debug_info.append(f"Rule {rule_idx}: Output variable '{output_var}' not found. Available: {list(outputs.keys())}")
                        continue

                    if term not in outputs[output_var].terms:
                        self.debug_info.append(f"Rule {rule_idx}: Term '{term}' not found for '{output_var}'. Available: {list(outputs[output_var].terms)}")
                        continue

                    fuzzy_consequent.append(outputs[output_var][term])

                # Create rule only if we have both antecedent and consequent
                if fuzzy_antecedent is not None and fuzzy_consequent:
                    rules.append(ctrl.Rule(fuzzy_antecedent, fuzzy_consequent))
                    self.debug_info.append(f"Rule {rule_idx}: Successfully added")
                else:
                    self.debug_info.append(f"Rule {rule_idx}: Could not create rule (missing parts)")

            except Exception as e:
                error_msg = f"Rule {rule_idx}: Error parsing rule - {str(e)}"
                self.debug_info.append(error_msg)
                print(error_msg)
                continue

        # Create control system
        if rules:
            control_system = ctrl.ControlSystem(rules)
            return ctrl.ControlSystemSimulation(control_system), inputs, outputs
        else:
            print("Warning: No valid rules created for fuzzy system!")
            return None, inputs, outputs

    def compute(self, inputs):
        """Perform fuzzy inference and return linguistic terms for all outputs"""
        if self.system is None:
            print("Fuzzy system not initialized - returning default values")
            results = {}
            for output_name in self.outputs.keys():
                results[output_name] = {
                    'crisp_value': None,
                    'linguistic_term': 'unknown',
                    'membership': 0
                }
            return results

        # Set input values
        for var_name, value in inputs.items():
            if var_name in self.inputs:
                self.system.input[var_name] = value

        try:
            # Compute the result
            self.system.compute()
        except Exception as e:
            error_msg = f"Fuzzy computation error: {e}"
            print(error_msg)
            self.debug_info.append(error_msg)

            # If computation failed, return 'unknown' for all outputs
            results = {}
            for output_name in self.outputs.keys():
                results[output_name] = {
                    'crisp_value': None,
                    'linguistic_term': 'unknown',
                    'membership': 0
                }
            return results

        # Get results for all outputs
        results = {}
        for output_name, output_var in self.outputs.items():
            try:
                crisp_output = self.system.output[output_name]

                # Determine linguistic term with highest membership
                max_membership = -1
                linguistic_term = "unknown"

                for term in output_var.terms:
                    membership = fuzz.interp_membership(
                        output_var.universe,
                        output_var[term].mf,
                        crisp_output
                    )

                    if membership > max_membership:
                        max_membership = membership
                        linguistic_term = term

                results[output_name] = {
                    'crisp_value': crisp_output,
                    'linguistic_term': linguistic_term,
                    'membership': max_membership
                }
            except KeyError:
                error_msg = f"Output '{output_name}' not found in results"
                print(error_msg)
                self.debug_info.append(error_msg)
                results[output_name] = {
                    'crisp_value': None,
                    'linguistic_term': 'unknown',
                    'membership': 0
                }
            except Exception as e:
                error_msg = f"Error processing output '{output_name}': {str(e)}"
                print(error_msg)
                self.debug_info.append(error_msg)
                results[output_name] = {
                    'crisp_value': None,
                    'linguistic_term': 'unknown',
                    'membership': 0
                }

        return results

    def print_debug_info(self):
        """Print debug information about rule parsing"""
        print("\nFuzzy System Debug Information:")
        for info in self.debug_info:
            print(f" - {info}")


if __name__ == "__main__":
    # Configuration
    data_path = "data.csv"
    x_columns = ['T1', 'T2', 'H1']
    y_column = 'H2'

    # Fuzzy variables configuration with two outputs
    fuzzy_vars_config = {
        'T1': {
            'min': 0,
            'max': 100,
            'terms': 3,
            'shape': 'gaussian'
        },
        'T2': {
            'min': 0,
            'max': 100,
            'terms': 3,
            'shape': 'gaussian'
        },
        'H1': {
            'min': 0,
            'max': 100,
            'terms': 5,
            'shape': 'triangular'
        },
        'H2': {
            'min': 0,
            'max': 100,
            'terms': 5,
            'shape': 'triangular'
        },
        'output_comfort': {
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
        },
        'output_risk': {
            'terms': {
                'none': {
                    'shape': 'triangular',
                    'params': [0, 0, 3]
                },
                'low': {
                    'shape': 'triangular',
                    'params': [0, 3, 6]
                },
                'medium': {
                    'shape': 'gaussian',
                    'params': [5, 1.0]
                },
                'high': {
                    'shape': 'triangular',
                    'params': [4, 7, 10]
                },
                'critical': {
                    'shape': 'triangular',
                    'params': [7, 10, 10]
                }
            }
        }
    }

    # CORRECTED fuzzy rules with proper spacing
    fuzzy_rules = [
        "If T1 is level_0 and T2 is level_0 and H1 is level_0 and H2 is level_0 then output_comfort is low and output_risk is none",
        "If T1 is level_0 and T2 is level_1 and H1 is level_1 and H2 is level_1 then output_comfort is low",
        "If T1 is level_1 and T2 is level_1 and H1 is level_2 and H2 is level_2 then output_comfort is medium",
        "If T1 is level_1 and T2 is level_2 and H1 is level_3 and H2 is level_3 then output_comfort is medium",
        "If T1 is level_2 and T2 is level_2 and H1 is level_4 and H2 is level_4 then output_comfort is high",
        "If T1 is level_0 and T2 is level_2 and H1 is level_0 and H2 is level_4 then output_risk is high",
        "If T1 is level_1 and T2 is level_1 and H1 is level_2 and H2 is level_3 then output_risk is medium",
        "If T1 is level_0 and T2 is level_1 and H1 is level_0 and H2 is level_1 then output_risk is low",
        "If T1 is level_2 and T2 is level_1 and H1 is level_4 and H2 is level_3 then output_risk is medium",
        "If T1 is level_1 and T2 is level_2 and H1 is level_3 and H2 is level_4 then output_risk is high",
        "If T1 is level_1 and T2 is level_1 and H1 is level_2 and H2 is level_2 then output_risk is low"
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
    fuzzy_results = fuzzy_module.compute(fuzzy_inputs)

    # Print debug information
    fuzzy_module.print_debug_info()

    # Print results for all outputs
    for output_name, result in fuzzy_results.items():
        print(f"\n{output_name.replace('_', ' ').title()}:")
        print(f"  Crisp value: {result['crisp_value'] if result['crisp_value'] is not None else 'N/A'}")
        print(f"  Linguistic term: {result['linguistic_term']}")
        print(f"  Membership: {result['membership']:.2f}")