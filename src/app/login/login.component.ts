import { Component, inject } from '@angular/core';
import { AuthService } from '../services/auth.service';

@Component({
  selector: 'app-login',
  imports: [],
  templateUrl: './login.component.html',
  styles: ``,
})
export class LoginComponent {
  private _authService = inject(AuthService);

  async login() {
    const authResponse = await this._authService.signIn();
  }
}
