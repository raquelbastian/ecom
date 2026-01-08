import SearchClient from './SearchClient';
import { Suspense } from 'react';

export default function SearchPage() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <SearchClient />
    </Suspense>
  );
}
