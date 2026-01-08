'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function SearchBar() {
  const [query, setQuery] = useState('');
  const router = useRouter();

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      router.push(`/search?query=${encodeURIComponent(query.trim())}`);
    }
  };

  return (
    <form onSubmit={handleSearch} className="flex items-center w-full max-w-2xl mx-auto">
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search for products..."
        className="flex-grow p-2 text-lg border-2 border-gray-300 rounded-l-md focus:outline-none focus:border-blue-500"
      />
      <button
        type="submit"
        className="p-2 text-lg text-white bg-blue-500 rounded-r-md hover:bg-blue-600 focus:outline-none"
      >
        Search
      </button>
    </form>
  );
}
