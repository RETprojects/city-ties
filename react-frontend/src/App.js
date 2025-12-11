// thanks to Dipal Bhavsar: https://www.bacancytechnology.com/blog/react-with-python

import logo from './logo.svg';
import './App.css';

import React, { useState, useEffect } from 'react';
import axios from 'axios';

// https://react.dev/learn/sharing-state-between-components
function filterItems(items, query) {
  query = query.toLowerCase();
  return items.filter(item =>
    item.name.split(' ').some(word =>
      word.toLowerCase().startsWith(query)
    )
  );
}

const App = () => {
    // thanks to Abdelhakim Smyaj: https://stackoverflow.com/a/79486650

    const [data, setData] = useState(null);

    useEffect(() => {
        fetch('/api/data')
        .then(response => response.json())
        .then(data => setData(data));
    }, []);

    return (
        <div>
        <h1>City Ties</h1>
        {data && <p>{data.message}</p>}
        </div>
    );
};

export default App;
