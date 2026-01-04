import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET(req: NextRequest) {
  try {
    const url = new URL(req.url);
    const product_id = url.searchParams.get('product_id');
    if (!product_id) return NextResponse.json({ error: 'product_id required' }, { status: 400 });

    const datasetDir = path.resolve(process.cwd(), 'app', 'dataset', 'recs_cache');
    const filePath = path.join(datasetDir, `${product_id}.json`);

    if (!fs.existsSync(filePath)) {
      return NextResponse.json({ found: false }, { status: 404 });
    }

    const raw = await fs.promises.readFile(filePath, 'utf-8');
    const payload = JSON.parse(raw);
    return NextResponse.json({ found: true, ...payload });
  } catch (err: any) {
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}