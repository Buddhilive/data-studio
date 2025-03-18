import { inject, Injectable } from '@angular/core';
import {
  AuthChangeEvent,
  Session,
  SupabaseClient,
  User,
  createClient,
} from '@supabase/supabase-js';
import { environment } from '../../environments/environment.development';
import { Router } from '@angular/router';

@Injectable({
  providedIn: 'root',
})
export class AuthService {
  private supabaseClient!: SupabaseClient;
  private user: User | undefined;
  private router = inject(Router);
  constructor() {
    this.supabaseClient = createClient(
      environment.supabaseUrl,
      environment.supabaseKey
    );

    this.supabaseClient.auth.onAuthStateChange((event, session) => {
      this.handleAuthSession(event, session);
    });
  }

  get isLoggedIn(): boolean {
    if (this.user) return true;
    return false;
  }
  async signIn() {
    await this.supabaseClient.auth.signInWithOAuth({
      provider: 'google',
    });
  }

  async signOut() {
    await this.supabaseClient.auth.signOut();
  }

  private handleAuthSession(event: AuthChangeEvent, session: Session | null) {
    console.log('Auth state', event, session);
    this.user = session?.user;
    if (this.user) this.router.navigate(['/']);
  }
}
