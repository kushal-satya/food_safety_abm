#!/bin/bash

# Script to update the results/index.html file

# Create results directory if it doesn't exist
mkdir -p results

# Create or update the index.html file
cat > results/index.html << 'EOL'
<!DOCTYPE html>
<html>
<head>
    <title>Farmer Risk Control Behaviors - Simulation Results</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            color: #333;
        }
        h1 {
            color: #0056b3;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }
        .scenario-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .scenario-card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 20px;
            transition: transform 0.3s ease;
        }
        .scenario-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .scenario-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #0056b3;
        }
        .scenario-description {
            margin-bottom: 15px;
            font-size: 14px;
            color: #666;
        }
        .scenario-link {
            display: inline-block;
            background-color: #0056b3;
            color: white;
            padding: 8px 16px;
            border-radius: 4px;
            text-decoration: none;
            font-size: 14px;
            transition: background-color 0.3s ease;
        }
        .scenario-link:hover {
            background-color: #003d82;
        }
    </style>
</head>
<body>
    <h1>Farmer Risk Control Behaviors - Simulation Results</h1>
    
    <div class="scenario-grid">
EOL

# Find all directories in results and add cards for them
for DIR in results/*/; do
    # Skip if not a directory or hidden directory
    if [ ! -d "$DIR" ] || [[ "$(basename "$DIR")" == .* ]]; then
        continue
    fi
    
    DIRNAME=$(basename "$DIR")
    # Create a nicer title
    TITLE=$(echo $DIRNAME | tr '_' ' ' | sed 's/\b\(.\)/\u\1/g')
    
    # Extract description from parameters.txt if available
    DESCRIPTION="Simulation scenario"
    if [ -f "$DIR/parameters.txt" ]; then
        FARMER_COUNT=$(grep "Number of farmers" "$DIR/parameters.txt" | head -n 1)
        TESTING=$(grep "Testing probabilities" "$DIR/parameters.txt" | head -n 1)
        
        if [ ! -z "$FARMER_COUNT" ] || [ ! -z "$TESTING" ]; then
            DESCRIPTION="$FARMER_COUNT $TESTING"
        fi
    fi
    
    # Add card to the HTML
    cat >> results/index.html << EOL
        <div class="scenario-card">
            <div class="scenario-title">$TITLE</div>
            <div class="scenario-description">
                $DESCRIPTION
            </div>
            <a href="$DIRNAME/report.html" class="scenario-link">View Results</a>
        </div>
EOL
done

# Close the HTML file
cat >> results/index.html << 'EOL'
    </div>
</body>
</html>
EOL

echo "Updated index.html with $(ls -l results/ | grep -c '^d') simulation directories"