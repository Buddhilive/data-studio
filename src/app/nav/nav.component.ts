import { Component, inject } from '@angular/core';
import { AuthService } from '../services/auth.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-nav',
  imports: [],
  templateUrl: './nav.component.html',
  styles: ``,
})
export class NavComponent {
  private _authService = inject(AuthService);
  private router = inject(Router);

  logout() {
    this._authService.signOut().then(() => {
      this.router.navigate(['/login']);
    }).catch((err) => alert(err));
  }
}
