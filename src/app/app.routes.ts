import { Routes } from '@angular/router';
import { DashboardComponent } from './dashboard/dashboard.component';
import { LoginComponent } from './login/login.component';
import { TranslationsComponent } from './translations/translations.component';
import { authGuard } from './auth.guard';

export const routes: Routes = [
  {
    path: '',
    canActivate: [authGuard],
    component: DashboardComponent,
  },
  {
    path: 'login',
    component: LoginComponent,
  },
  {
    path: 'translations',
    canActivate: [authGuard],
    component: TranslationsComponent,
  },
];
